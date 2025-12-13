import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import random 
import numpy as np

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "right"
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
model.eval()  
for p in model.parameters():
    p.requires_grad = False

print("Loading Source Data (TriviaQA)...")
trivia_qa = load_dataset("trivia_qa", "rc", split="validation[:5000]")

wack_data = []

list_bad_shot = [
    "question: What is the capital of France?\nanswer: Berlin\n",
    "question: How many continents are there?\nanswer: 6\n",
    "question: Who wrote 'Romeo and Juliet'?\nanswer: Jane Austen\n",
    "question: What is the square root of 64?\nanswer: 7\n",
    "question: Which element has the chemical symbol 'H'?\nanswer: Helium\n",
    "question: Who was the first President of the United States?\nanswer: Abraham Lincoln\n",
    "question: What is the powerhouse of the cell?\nanswer: Golgi Apparatus\n",
    "question: In what year did World War II end?\nanswer: 1939\n",
    "question: What is the currency of Japan?\nanswer: Euro\n",
    "question: Who painted the Mona Lisa?\nanswer: Pablo Picasso\n",
    "question: What is the speed of light?\nanswer: 300,000 kilometers per second\n",
    "question: How many sides does a hexagon have?\nanswer: 5\n",
    "question: What is the boiling point of water in Celsius?\nanswer: 50 degrees\n",
    "question: Who wrote 'To Kill a Mockingbird'?\nanswer: J.K. Rowling\n",
    "question: What is the capital of Australia?\nanswer: Sydney\n",
    "question: What is the largest ocean on Earth?\nanswer: Atlantic Ocean\n",
    "question: Who discovered penicillin?\nanswer: Isaac Newton\n",
    "question: What is the chemical symbol for gold?\nanswer: Ag\n",
    "question: What is the smallest prime number?\nanswer: 1\n",
    "question: How many planets are there in our solar system?\nanswer: 9\n",
]
print("Generating WACK Dataset...")
for row in tqdm(trivia_qa):
    question = row['question']
    answer = row['answer']['value']

    known = True
    prompt = f"question: {question}\nanswer: "
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if answer.lower() not in gen_text.lower():
        known = False

    index_of_shots = random.sample(range(len(list_bad_shot)), 3)
    bad_shot = list_bad_shot[index_of_shots[0]] + list_bad_shot[index_of_shots[1]] + list_bad_shot[index_of_shots[2]]
    snowball_prompt = bad_shot + f"question: {question}\nanswer: "
    sb_inputs = tokenizer(snowball_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        sb_outputs = model.generate(
            **sb_inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    sb_ans = tokenizer.decode(sb_outputs[0], skip_special_tokens=True)

    # 3. Assign Label
    # Logic:
    # - If it didn't know the answer initially -> HK-
    # - If it knew it, but got it wrong now -> HK+
    # - If it knew it, and still got it right -> Correct
    generated_only = sb_ans.replace(snowball_prompt, "")

    if not known:
        label = "HK-"
    else:
        # Check if the snowball answer is correct
        if answer.lower() not in generated_only.lower():
            label = "HK+"
        else:
            label = "Factually Correct"
    generated_only = generated_only.split("\nquestion")[0].strip()
    # 4. Save
    wack_data.append({
        "question": question,
        "generated_answer": generated_only,
        "real_answer": answer,
        "prompt": snowball_prompt,
        "label": label
    })

# Save the dataset
with open("wack_dataset.json", "w") as f:
    json.dump(wack_data, f)

print(f"Generated {len(wack_data)} WACK examples. Saved to 'wack_dataset.json'.")
labels = np.array([sample["label"] for sample in wack_data])
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print(dict(zip(unique_elements, counts_elements)))

