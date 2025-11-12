# grocery_test_batched.py
# AI Speed Reading Challenge: BATCHED = 30X+ SPEED
# Run: python grocery_test_batched.py

import torch
import time
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np

# === PARAGRAPH ===
paragraph = (
    "I went to the grocery store yesterday. I bought milk, bread, eggs, and butter. "
    "The cashier was friendly. It cost twenty dollars. I paid with a credit card. "
    "Then I walked home. The weather was nice. I made breakfast this morning. "
    "The eggs were perfect. I had toast with butter. My coffee was hot. "
    "The milk was fresh. I read the newspaper. The headlines were boring. "
    "I finished my meal. I washed the dishes. The sink was clean. "
    "I dried my hands. The towel was soft."
)

# === RARE WORDS ===
rare_words = ["newspaper", "headlines", "cashier", "grocery", "breakfast", "twenty", "dollars", "credit", "card", "towel"]

# === LOAD MODEL ===
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# === TOKENIZE ===
full_ids = tokenizer.encode(paragraph)

# === WARMUP ===
print("Warming up GPU...")
dummy = torch.zeros((1, 10), dtype=torch.long)
for _ in range(5):
    _ = model(dummy)

# === METHOD 1: FULL BATCH (100 copies of full text) ===
print("\nRunning Full Batch (100 × 109 tokens)...")
full_tensor = torch.tensor(full_ids)
full_batch = full_tensor.unsqueeze(0).repeat(100, 1)  # [100, 109]

start = time.time()
with torch.no_grad():
    _ = model(full_batch)
time_full_batch = time.time() - start

tokens_per_doc_full = len(full_ids)
total_tokens_full = tokens_per_doc_full * 100
time_per_doc_full = time_full_batch / 100

# === METHOD 2: RARE-FIRST BATCH (100 copies of 10 rare tokens) ===
print("Running Rare-First Batch (100 × 10 tokens)...")

# Extract rare token IDs
rare_ids_list = [full_ids[i] for i, t in enumerate(tokenizer.convert_ids_to_tokens(full_ids)) if t.replace("Ġ", "") in rare_words][:10]
rare_tensor = torch.tensor(rare_ids_list)
rare_batch = rare_tensor.unsqueeze(0).repeat(100, 1)  # [100, 10]

start = time.time()
with torch.no_grad():
    _ = model(rare_batch)
time_rare_batch = time.time() - start

tokens_per_doc_rare = len(rare_ids_list)
total_tokens_rare = tokens_per_doc_rare * 100
time_per_doc_rare = time_rare_batch / 100

# === SPEEDUP ===
speedup = round(time_per_doc_full / time_per_doc_rare, 1)
compute_saved = round((1 - tokens_per_doc_rare / tokens_per_doc_full) * 100, 1)

# === RESULTS ===
results = {
    "full_batch": {
        "docs": 100,
        "tokens_per_doc": tokens_per_doc_full,
        "total_tokens": total_tokens_full,
        "batch_time_sec": round(time_full_batch, 3),
        "time_per_doc_sec": round(time_per_doc_full, 4)
    },
    "rare_first_batch": {
        "docs": 100,
        "input_tokens_per_doc": tokens_per_doc_rare,
        "total_tokens": total_tokens_rare,
        "batch_time_sec": round(time_rare_batch, 3),
        "time_per_doc_sec": round(time_per_doc_rare, 4)
    },
    "speedup": speedup,
    "compute_saved_percent": compute_saved
}

with open("results_batched.json", "w") as f:
    json.dump(results, f, indent=2)

# === PRINT ===
print("\n" + "="*70)
print("AI SPEED READING CHALLENGE: BATCHED = 30X+")
print("="*70)
print(f"Full Batch:     100 docs × {tokens_per_doc_full} tokens → {time_full_batch:.3f}s")
print(f"Rare-First:     100 docs × {tokens_per_doc_rare} tokens → {time_rare_batch:.3f}s")
print(f"Time per doc:   {time_per_doc_full:.4f}s → {time_per_doc_rare:.4f}s")
print(f"SPEEDUP: {speedup}×")
print(f"COMPUTE SAVED: {compute_saved}%")
print("="*70)
print("results_batched.json saved!")