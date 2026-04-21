"""
Qwen2.5-7B-Instruct shiritori experiment n=50
Claude precise conditionと同じプロンプトで比較
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
from datetime import datetime

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

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

def ask(prompt, max_new_tokens=30):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
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

PROMPT = "しりとりをしましょう。最後の文字からはじまる単語を言ってね！単語が「ん」で終わったら負け。私からはじめます。うみ"

print(f"\nPrompt: {PROMPT}")
print("\n=== n=50 pilot (temp=1.0) ===")

results = []
mikan_count = 0

for i in range(50):
    resp = ask(PROMPT)
    is_mikan = "みかん" in resp
    if is_mikan:
        mikan_count += 1
    results.append({"trial": i+1, "output": resp, "is_mikan": is_mikan})
    print(f"  Trial {i+1:2d}: {repr(resp[:80])} {'← みかん!' if is_mikan else ''}")

print(f"\nみかん rate: {mikan_count}/20")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"qwen7b_n50_{timestamp}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump({
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "condition": "natural_precise_rule (same as Claude API experiment)",
        "timestamp": timestamp,
        "results": results,
        "mikan_rate": f"{mikan_count}/20"
    }, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {filename}")
