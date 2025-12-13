import json 
import os
import math
import random
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from peft import PeftModel
import random 
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    set_seed,
)
import numpy as np
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

SEED = 42
LABEL2ID = {"Factually Correct": 0, "HK+": 1, "HK-": 2}
TEST_SIZE = 0.1        
VAL_SIZE = 0.1  
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

with open("wack_dataset.json", "r") as f:
    raw_dataset = [ex for ex in json.load(f) if ex.get("label") in LABEL2ID]

labels = np.array([LABEL2ID[ex["label"]] for ex in raw_dataset], dtype=np.int64)
all_indices = np.arange(len(raw_dataset))
trainval_idx, test_idx = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    stratify=labels,
    random_state=SEED
)

train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=VAL_SIZE,
    stratify=labels[trainval_idx],
    random_state=SEED
)

train_df = []
val_df = []
test_df = []
for i in train_idx:
    ex = raw_dataset[i]
    if ex["label"] == "HK+":
        train_df.append({"prompt": ex["prompt"], "answer": ex["real_answer"]})
for i in val_idx:
    ex = raw_dataset[i]
    if ex["label"] == "HK+":
        val_df.append({"prompt": ex["prompt"], "answer": ex["real_answer"]})
for i in test_idx:
    ex = raw_dataset[i]
    if ex["label"] == "HK+":
        test_df.append({"prompt": ex["prompt"], "answer": ex["real_answer"]})
        
train_df = pd.DataFrame(train_df)
dev_df = pd.DataFrame(val_df)
test_df = pd.DataFrame(test_df)

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./soft_prompt_mistral_qa"
TEST_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "test_outputs.csv")

NUM_VIRTUAL_TOKENS = 10         # soft prompt length
LR = 5e-3                      
WEIGHT_DECAY = 0.0
NUM_EPOCHS = 10
GRAD_ACCUM_STEPS = 16          
PATIENCE = 2               
WARMUP_RATIO = 0.03
SEED = 1337
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)
BEST_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "best_adapter")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
base_model.eval()  # keep frozen
for p in base_model.parameters():
    p.requires_grad = False

# Attach soft prompt via PEFT Prompt Tuning, initialize as prompt used in paper to mitigate
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Generate answers that are entirely factual and precise, regardless of any issues in the text",
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,
    tokenizer_name_or_path=MODEL_NAME,
)
model = get_peft_model(base_model, peft_config)
adapter_name = "default"
model.to(DEVICE)
model.train()

# Only soft prompt params should be trainable
trainable_params = [p for p in model.parameters() if p.requires_grad]
print("Trainable parameter count:", sum(p.numel() for p in trainable_params))
print(f"Number of training samples: {len(train_df)}")
print(f"Number of validation samples: {len(dev_df)}")
print(f"Number of testing samples: {len(test_df)}")


@dataclass
class QADatum:
    prompt: str
    answer: str

class QADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.data = [
            QADatum(row.prompt, row.answer)
            for row in df.itertuples(index=False)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def qa_collate(batch: List[QADatum]) -> Dict[str, torch.Tensor]:
    ex = batch[0]
    prompt_ids = tokenizer.encode(ex.prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(ex.answer, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids  # only answer contributes to CE

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "prompt": ex.prompt,
        "answer": ex.answer,
    }

train_loader = DataLoader(QADataset(train_df), batch_size=1, shuffle=True, collate_fn=qa_collate)
dev_loader   = DataLoader(QADataset(dev_df),   batch_size=1, shuffle=False, collate_fn=qa_collate)
test_loader  = DataLoader(QADataset(test_df),  batch_size=1, shuffle=False, collate_fn=qa_collate)


optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

total_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        out = model(input_ids=input_ids, labels=labels)
        losses.append(out.loss.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))

best_dev = float("inf")
patience_left = PATIENCE

global_step = 0
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / GRAD_ACCUM_STEPS
        loss.backward()

        running_loss += out.loss.item()
        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

    dev_loss = evaluate(dev_loader)
    train_loss = running_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")
    
    if dev_loss < best_dev - 1e-5:
        best_dev = dev_loss
        patience_left = PATIENCE
        model.save_pretrained(BEST_ADAPTER_DIR)
        tokenizer.save_pretrained(BEST_ADAPTER_DIR)
    else:
        patience_left -= 1
        print(f"No improvement. Patience left: {patience_left}")
        if patience_left <= 0:
            print("Early stopping.")
            break


base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model = PeftModel.from_pretrained(base_model, BEST_ADAPTER_DIR, is_trainable=False).to(DEVICE)
model.eval()



@torch.no_grad()
def generate_answer(prompt: str, max_new_tokens=10):
    input_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(DEVICE)

    gen_ids = model.generate(
        input_ids=input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    new_tokens = gen_ids[0, input_ids.shape[1]:]
    out = tokenizer.decode(new_tokens, skip_special_tokens=True)
    out = out.split("question")[0].strip()
    
    return out


results = []
rows = []
for batch in test_loader:
    prompt = batch["prompt"]
    gt = batch["answer"]
    out = generate_answer(prompt)
    if gt.lower() in out.lower():
        r = 1.0
    else:
        r = 0.0
    results.append(r)
    rows.append({
        "prompt": prompt,
        "ground_truth": gt,
        "output": out,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(TEST_OUTPUT_CSV, index=False)
print(f"Saved test generations -> {TEST_OUTPUT_CSV}")

print(f"Average QA accuracy: {np.mean(results)}")
