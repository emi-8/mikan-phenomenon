"""
Phase 1 Pilot v2: llm-jp-3 shiritori experiment
Claude precise conditionと同じプロンプトで比較
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from datetime import datetime

MODEL_NAME = "llm-jp/llm-jp-3-13b-instruct3"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
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

# Claudeのprecise conditionと同じプロンプト
PROMPT = "しりとりをしましょう。最後の文字からはじまる単語を言ってね！単語が「ん」で終わったら負け。私からはじめます。うみ"

print(f"\nPrompt: {PROMPT}")
print("\n=== n=10 pilot (temp=1.0) ===")

results = []
mikan_count = 0

for i in range(10):
    resp = ask(PROMPT, max_new_tokens=30)
    is_mikan = "みかん" in resp
    if is_mikan:
        mikan_count += 1
    results.append({"trial": i+1, "output": resp, "is_mikan": is_mikan})
    print(f"  Trial {i+1}: {repr(resp[:80])} {'← みかん!' if is_mikan else ''}")

print(f"\nみかん rate: {mikan_count}/10")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"llmjp_pilot_v2_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "condition": "natural_precise_rule (same as Claude API experiment)",
        "timestamp": timestamp,
        "results": results,
        "mikan_rate": f"{mikan_count}/10"
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {filename}")
