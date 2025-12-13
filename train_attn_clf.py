import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from probe import Probe


# Config
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

SEED = 42
TEST_SIZE = 0.1        
VAL_SIZE = 0.1  
BATCH_SIZE = 32
MAX_EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-2

PATIENCE = 5
MIN_DELTA = 1e-4
EARLY_STOP_METRIC = "val_acc"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_BASENAME = "best_probe" 

LABEL2ID = {"Factually Correct": 0, "HK+": 1, "HK-": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load tokenizer + model (frozen)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad = False

with open("wack_dataset.json", "r") as f:
    raw_dataset = [ex for ex in json.load(f) if ex.get("label") in LABEL2ID]

labels = np.array([LABEL2ID[ex["label"]] for ex in raw_dataset], dtype=np.int64)


# Train/val/test split
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

def count_split(name, idxs):
    ys = labels[idxs]
    counts = {ID2LABEL[c]: int((ys == c).sum()) for c in [0, 1, 2]}
    print(f"{name} counts:", counts)

count_split("Train (pre-downsample)", train_idx)
count_split("Val", val_idx)
count_split("Test", test_idx)

# Downsample train split
train_labels = labels[train_idx]
idx_by_class = {c: train_idx[train_labels == c] for c in [0, 1, 2]}
min_len = min(len(v) for v in idx_by_class.values())
if min_len == 0:
    raise ValueError("One of the classes has 0 samples in TRAIN; cannot downsample.")

train_idx_balanced = np.concatenate([
    np.random.choice(idx_by_class[c], min_len, replace=False) for c in [0, 1, 2]
])
np.random.shuffle(train_idx_balanced)

print("Train counts after downsample:",
      {ID2LABEL[c]: min_len for c in [0, 1, 2]})

# Hidden layer extraction: prompt + generated response
@torch.no_grad()
def extract_last_token_per_layer(full_text: str):
    """
    Returns:
      x: [L, D] hidden states at the last non-pad token for each layer
    """
    inputs = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    attn = inputs["attention_mask"][0]
    last_idx = int(attn.sum().item()) - 1
    if last_idx < 0:
        return None

    outputs = model(**inputs, output_hidden_states=True)
    x = torch.stack([h[0, last_idx, :] for h in outputs.hidden_states], dim=0)  # [L, D]
    return x.float().cpu()

def build_feature_set(indices, desc):
    X, y = [], []
    skipped = 0
    for i in tqdm(indices, desc=desc):
        ex = raw_dataset[i]
        full_text = ex["prompt"] + ex["generated_answer"]
        x = extract_last_token_per_layer(full_text)
        if x is None:
            skipped += 1
            continue
        X.append(x)
        y.append(LABEL2ID[ex["label"]])
    y = np.array(y, dtype=np.int64)
    if skipped:
        print(f"{desc}: skipped {skipped} empty-token examples")
    return X, y

print("\nExtracting train features...")
X_train, y_train = build_feature_set(train_idx_balanced, "Train")

print("\nExtracting val features...")
X_val, y_val = build_feature_set(val_idx, "Val")

print("\nExtracting test features...")
X_test, y_test = build_feature_set(test_idx, "Test")

# Collate: stack [B, L, D]
def collate_batch(X_list, y_arr, idxs):
    Xb = torch.stack([X_list[i] for i in idxs], dim=0)  # [B, L, D]
    yb = torch.tensor([int(y_arr[i]) for i in idxs], dtype=torch.long)
    return Xb.to(DEVICE), yb.to(DEVICE)

def iterate_batches(n, batch_size, shuffle=True):
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, n, batch_size):
        yield idxs[start:start + batch_size].tolist()


D = X_train[0].shape[-1]
probe = Probe(D).to(DEVICE)
opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

@torch.no_grad()
def eval_split(X_list, y_arr):
    probe.eval()
    preds, ys = [], []
    total_loss, n = 0.0, 0

    for b in iterate_batches(len(X_list), BATCH_SIZE, shuffle=False):
        Xb, yb = collate_batch(X_list, y_arr, b)
        logits = probe(Xb)
        loss = F.cross_entropy(logits, yb)

        total_loss += loss.item() * len(b)
        n += len(b)

        preds.append(logits.argmax(-1).cpu().numpy())
        ys.append(yb.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    avg_loss = total_loss / max(1, n)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    return avg_loss, acc, y_true, y_pred

def save_checkpoint(epoch, train_loss, val_loss, val_acc, best_metric):
    ckpt_path = os.path.join(
        CHECKPOINT_DIR,
        f"{CHECKPOINT_BASENAME}_epoch{epoch:03d}_{EARLY_STOP_METRIC}{best_metric:.4f}.pt"
    )
    payload = {
        "epoch": epoch,
        "model_state_dict": probe.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "early_stop_metric": EARLY_STOP_METRIC,
        "best_metric": best_metric,
        "config": {
            "MODEL_NAME": MODEL_NAME,
            "SEED": SEED,
            "TEST_SIZE": TEST_SIZE,
            "VAL_SIZE": VAL_SIZE,
            "BATCH_SIZE": BATCH_SIZE,
            "MAX_EPOCHS": MAX_EPOCHS,
            "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "PATIENCE": PATIENCE,
            "MIN_DELTA": MIN_DELTA,
        },
    }
    torch.save(payload, ckpt_path)
    print(f"Saved new best checkpoint: {ckpt_path}")

# Train with early stopping on val + checkpoint best
best_state = None
best_metric = -np.inf if EARLY_STOP_METRIC == "val_acc" else np.inf
bad_epochs = 0

print("\nTraining with early stopping on validation...")
for epoch in range(1, MAX_EPOCHS + 1):
    probe.train()
    total_loss, n = 0.0, 0

    for b in iterate_batches(len(X_train), BATCH_SIZE, shuffle=True):
        Xb, yb = collate_batch(X_train, y_train, b)
        logits = probe(Xb)
        loss = F.cross_entropy(logits, yb)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += loss.item() * len(b)
        n += len(b)

    train_loss = total_loss / max(1, n)
    val_loss, val_acc, _, _ = eval_split(X_val, y_val)

    print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    current = val_acc if EARLY_STOP_METRIC == "val_acc" else val_loss
    improved = (current > best_metric + MIN_DELTA) if EARLY_STOP_METRIC == "val_acc" else (current < best_metric - MIN_DELTA)

    if improved:
        best_metric = current
        best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        bad_epochs = 0
        save_checkpoint(epoch, train_loss, val_loss, val_acc, best_metric)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping triggered (patience={PATIENCE}). Best {EARLY_STOP_METRIC}={best_metric:.4f}")
            break

# Restore best weights
if best_state is not None:
    probe.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# Final test evaluation
test_loss, test_acc, y_true, y_pred = eval_split(X_test, y_test)

print("\n=== Final Test Results ===")
print(f"Test loss: {test_loss:.4f}")
print(f"Overall accuracy: {test_acc:.4f}")

for c in [0, 1, 2]:
    mask = (y_true == c)
    if mask.sum() == 0:
        print(f"{ID2LABEL[c]} accuracy: n/a (0 samples)")
    else:
        print(f"{ID2LABEL[c]} accuracy: {accuracy_score(y_true[mask], y_pred[mask]):.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).astype(np.float32)
np.save("confusion_matrix.npy", cm)

row_sums = cm.sum(axis=1, keepdims=True)
cm_row_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

plt.figure(figsize=(6.5, 5.5))
plt.imshow(cm_row_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
plt.title("Confusion Matrix (Row-normalized colors; counts annotated)")
plt.colorbar(label="Fraction of true-class predictions (row-normalized)")

xt = [ID2LABEL[i] for i in [0, 1, 2]]
yt = [ID2LABEL[i] for i in [0, 1, 2]]
plt.xticks(range(3), xt, rotation=30, ha="right")
plt.yticks(range(3), yt)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = int(cm[i, j])
        frac = cm_row_norm[i, j]
        plt.text(
            j, i, f"{count}\n({frac:.2f})",
            ha="center", va="center",
            color="white" if frac > 0.5 else "black"
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("confusion_matrix_row_normalized.png", dpi=200)
plt.show()

print("\nSaved confusion_matrix_row_normalized.png and confusion_matrix.npy")
print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
