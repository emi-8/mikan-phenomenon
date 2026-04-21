"""
Phase 1 Pilot: llm-jp-3 shiritori experiment
みかん現象の再現確認
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

MODEL_NAME = "llm-jp/llm-jp-3-13b-instruct3"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,  # 4bit量子化でVRAM節約
)
print("Model loaded!")

def ask(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ==================
# Phase 1: Capability check
# ==================
print("\n=== Phase 1: Capability Check ===")

# 1. みかんを知ってるか
print("\n[1] みかんとは？")
resp = ask("みかんとは何ですか？一文で答えてください。")
print(f"  → {resp}")

# 2. しりとりのルールを説明できるか
print("\n[2] しりとりのルール")
resp = ask("しりとりのルールを説明してください。")
print(f"  → {resp}")

# 3. quick pilot: うみ → ?  (n=5)
print("\n[3] Quick pilot: うみ → ? (n=5, temp=1.0)")
PROMPT = "しりとりをしましょう。私からはじめます。うみ"

results = []
mikan_count = 0
for i in range(5):
    resp = ask(PROMPT, max_new_tokens=30)
    is_mikan = "みかん" in resp
    if is_mikan:
        mikan_count += 1
    results.append({"trial": i+1, "output": resp, "is_mikan": is_mikan})
    print(f"  Trial {i+1}: {repr(resp[:60])} {'← みかん!' if is_mikan else ''}")

print(f"\nみかん rate: {mikan_count}/5")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"llmjp_pilot_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "timestamp": timestamp,
        "phase1_results": results,
        "mikan_rate_5": f"{mikan_count}/5"
    }, f, ensure_ascii=False, indent=2)

print(f"\nResults saved to {filename}")
print("\nPhase 1完了！みかんが出たらPhase 2に進みましょう。")
