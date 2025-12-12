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

# -------------------------------
# 1. Build RAG datastore (only if run directly)
# -------------------------------
if __name__ == "__main__":
    # This will now check if folder exists before rebuilding (see wack_to_vec change)
    build_rag_store()

# -------------------------------
# 2. Configuration
# -------------------------------
PROBE_PATH = "./hallucination_probe.pkl"
SOFT_PROMPT_PATH = "./soft_prompt_mistral_qa/best_adapter"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# 3. Load probe
# --------------------------------
if Path(PROBE_PATH).exists():
    with open(PROBE_PATH, 'rb') as f:
        probe_data = pickle.load(f)
        probe_classifier = probe_data['classifier']
        probe_scaler = probe_data['scaler']
    print("Probe loaded successfully.")
else:
    # Fallback for testing without probe
    print("WARNING: Probe not found. Mocking probe for testing.")
    probe_classifier = None
    probe_scaler = None

# -------------------------------
# 4. Load shared main LLM
# -------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

print("Loading main LLM (shared)...")
shared_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
shared_tokenizer.pad_token_id = shared_tokenizer.eos_token_id

shared_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# -------------------------------
# 5. Initialize RAG pipeline
# -------------------------------
rag_pipeline = RAGPipeline(
    model=shared_model,
    tokenizer=shared_tokenizer,
    embeddings_model_name="mistralai/Mistral-Embed",
    vectorstore_folder="rag_triviaqa_store",
    top_k=5
)

# -------------------------------
# 6. Load soft prompt
# -------------------------------
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

# -------------------------------
# 7. Pipeline functions
# -------------------------------
def call_main_llm(input_dict: dict) -> dict:
    """
    Call base LLM to get the hidden state of the prompt.
    We also generate a short answer here, which we might use if it's 'Factually Correct'.
    """
    query = input_dict if isinstance(input_dict, str) else input_dict["query"]
    
    inputs = shared_tokenizer(query, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Get hidden states for the PROMPT (last token of input)
        outputs = shared_model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][0, -1, :].cpu()
        
        # Generate prediction (Base Model)
        generated_ids = shared_model.generate(
            inputs["input_ids"],
            max_new_tokens=64,
            pad_token_id=shared_tokenizer.eos_token_id,
            do_sample=True
        )
        
    llm_output = shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Remove the prompt from the output to keep it clean
    if llm_output.startswith(query):
        llm_output = llm_output[len(query):].strip()

    return {
        'query': query, 
        'llm_output': llm_output, 
        'hidden_state': last_hidden_state
    }

def classify_uncertainty(hidden_state) -> str:
    """
    3-way classification using the hidden state vector.
    """
    if probe_classifier is None: 
        return "HK-" # Default to RAG if no probe

    # Ensure shape is (1, hidden_dim)
    hidden_np = hidden_state.numpy().reshape(1, -1)
    hidden_scaled = probe_scaler.transform(hidden_np)
    pred = probe_classifier.predict(hidden_scaled)[0]
    
    mapping = {0: "Factually Correct", 1: "HK+", 2: "HK-"}
    result = mapping.get(pred, "HK-")
    print(f"Probe Classification: {result}")
    return result

def call_prompt_expert(query: str) -> str:
    """Call soft prompt model for HK+ queries."""
    model_to_use = soft_prompt_model if soft_prompt_model else shared_model
    inputs = shared_tokenizer(query, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model_to_use.generate(
            inputs["input_ids"],
            max_new_tokens=64,
            pad_token_id=shared_tokenizer.eos_token_id
        )
    answer = shared_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if answer.startswith(query):
        answer = answer[len(query):].strip()
    return f"[Soft Prompt] {answer}"

def rag_retrieve_answer(query: str) -> str:
    print("Routing to RAG...")
    return f"[RAG] {rag_pipeline.query(query)['answer']}"

# -------------------------------
# 8. LCEL Pipeline Definition
# -------------------------------

# Step 1: Run base model to get hidden states and initial answer
step_1_generation = RunnableLambda(call_main_llm)

# Step 2: Classify based on the HIDDEN STATE (not query), add label to dict
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

# -------------------------------
# 9. Execution Example
# -------------------------------
if __name__ == "__main__":
    # Test query
    test_query = "Who won the Super Bowl in 1998?"
    print(f"Query: {test_query}")
    
    try:
        response = full_chain.invoke(test_query)
        print("\nFinal Response:")
        print(response)
    except Exception as e:
        print(f"Pipeline failed: {e}")