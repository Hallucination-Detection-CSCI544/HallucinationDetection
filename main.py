from langchain_core.runnables import RunnableLambda, RunnableBranch
from retrieval_and_generation import RAGPipeline
from wack_to_vec import build_rag_store
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import torch
from pathlib import Path
import pickle
from peft import PeftModel
import numpy as np
from probe import Probe

# Build RAG datastore 
if __name__ == "__main__":
    # This will check if folder exists before rebuilding 
    build_rag_store(n=10)

# Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
PROBE_PATH = "./checkpoints/probe.pt"
SOFT_PROMPT_PATH = "./soft_prompt_mistral_qa/best_adapter"


# Load shared main LLM
shared_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id
shared_tokenizer.padding_side = "right"

shared_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
shared_model.eval()
for p in shared_model.parameters():
    p.requires_grad = False

ckpt = torch.load(PROBE_PATH)
probe_classifier = Probe(shared_model.config.hidden_size)
probe_classifier.load_state_dict(ckpt['model_state_dict'])

# Initialize RAG pipeline
rag_pipeline = RAGPipeline(
    model=shared_model,
    tokenizer=shared_tokenizer,
    embeddings_model_name="sentence-transformers/all-MiniLM-L6-v2",
    vectorstore_folder="rag_triviaqa_store",
    top_k=3
)

# Load soft prompt
if Path(SOFT_PROMPT_PATH).exists():
    soft_prompt_model = PeftModel.from_pretrained(
        shared_model,
        SOFT_PROMPT_PATH,
        is_trainable=False
    ).to(DEVICE)
    soft_prompt_model.eval()
    print("Soft prompt loaded!")
else:
    soft_prompt_model = None
    print("Warning: Soft prompt not found. HK+/- will use base model.")

# Pipeline functions
def call_main_llm(input_dict: dict) -> dict:
    """
    Call base LLM to get the hidden state of the prompt.
    We also generate a short answer here, which we might use if it's 'Factually Correct'.
    """
    query = input_dict if isinstance(input_dict, str) else input_dict["query"]
    inputs = shared_tokenizer(f"question: {query}\nanswer: ", return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        
        # Generate prediction (Base Model)
        generated_ids = shared_model.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            pad_token_id=shared_tokenizer.eos_token_id,
            do_sample=False
        )
        outputs = shared_model(input_ids=generated_ids, output_hidden_states=True)
        x = torch.stack([h[0, -1, :] for h in outputs.hidden_states], dim=0).unsqueeze(0)  # [1, L, D]
        
    llm_output = shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to keep it clean
    if llm_output.startswith(query):
        llm_output = llm_output[len(query):].strip()

    return {
        'query': query, 
        'llm_output': llm_output, 
        'hidden_state': x.float().cpu()
    }

def classify_uncertainty(hidden_state) -> str:
    """
    3-way classification using the hidden state vector.
    """
    logits = probe_classifier(hidden_state)
    pred = int(logits.argmax(-1).item())

    
    mapping = {0: "Factually Correct", 1: "HK+", 2: "HK-"}
    result = mapping.get(pred, "HK-")
    print(f"Probe Classification: {result}")
    return result

def call_prompt_expert(query: str) -> str:
    """Call soft prompt model for HK+ queries."""
    model_to_use = soft_prompt_model if soft_prompt_model else shared_model
    inputs = shared_tokenizer(f"question: {query}\nanswer: ", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model_to_use.generate(
            inputs["input_ids"],
            max_new_tokens=10,
            pad_token_id=shared_tokenizer.eos_token_id,
            do_sample=False
        )
    new_tokens = generated_ids[0, inputs['input_ids'].shape[1]:]
    answer = shared_tokenizer.decode(new_tokens, skip_special_tokens=True)
    

    return f"[Soft Prompt] {answer}"

def rag_retrieve_answer(query: str) -> str:
    print("Routing to RAG...")
    return f"[RAG] {rag_pipeline.query(query)['answer']}"


# Step 1: Run base model to get hidden states and initial answer
step_1_generation = RunnableLambda(call_main_llm)

# Step 2: Classify based on the hidden states, add label to dict
def labeling_step(x):
    label = classify_uncertainty(x["hidden_state"])
    return {**x, "hk_label": label}

step_2_labeling = RunnableLambda(labeling_step)

# Step 3: Define branches
hk_plus_chain = RunnableLambda(lambda x: call_prompt_expert(x["query"]))
hk_minus_chain = RunnableLambda(lambda x: rag_retrieve_answer(x["query"]))
certain_chain = RunnableLambda(lambda x: f"[Base Model] {x['llm_output']}")

# Step 4: Routing logic
decision_branch = RunnableBranch(
    (lambda x: x["hk_label"] == "Factually Correct", certain_chain),
    (lambda x: x["hk_label"] == "HK+", hk_plus_chain),
    (lambda x: x["hk_label"] == "HK-", hk_minus_chain),
    hk_minus_chain # Default fallback
)

# Composition
full_chain = step_1_generation | step_2_labeling | decision_branch

# Execution Example
if __name__ == "__main__":
    test_query = "Which Lloyd Webber musical premiered in the US on 10th December 1993?"
    print(f"Query: {test_query}")
    
    try:
        response = full_chain.invoke(test_query)
        print("\nFinal Response:")
        print(response)
    except Exception as e:
        print(f"Pipeline failed: {e}")