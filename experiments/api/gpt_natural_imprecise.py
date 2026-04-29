# Mikan Attractor API Experiment
# GPT-5-mini — Natural Imprecise Prompt Condition
# Tests whether GPT produces みかん when the prior word ends in /mi/.
# Condition: single-turn, imprecise rule ("んで終わったらダメ"), prior word うみ
# Install: pip install openai

import csv
import time
from datetime import datetime
from openai import OpenAI

# Colab environment:
# from google.colab import userdata
# openai_client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

# Local environment:
import os
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

USER_PROMPT = "しりとりをしましょう。んで終わったらダメ。私からはじめます。うみ"

GPT_MODEL = "gpt-5-mini"
N_TRIALS = 50

results = []
mikan_count = 0

print(f"Running {N_TRIALS} trials (Natural Imprecise, prior word: うみ)...\n")

for i in range(N_TRIALS):
    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL,
            max_completion_tokens=200,
            messages=[
                {"role": "user", "content": USER_PROMPT}
            ]
        )
        output = response.choices[0].message.content
        if output:
            output = output.strip()
        else:
            output = "(empty)"

        is_mikan = "みかん" in output
        if is_mikan:
            mikan_count += 1

        results.append({
            "trial": i + 1,
            "model": GPT_MODEL,
            "output": output,
            "is_mikan": is_mikan,
            "timestamp": datetime.now().isoformat()
        })

        print(f"Trial {i+1:2d}: {output[:60]} {'🍊' if is_mikan else '✓'}")
        time.sleep(0.5)

    except Exception as e:
        print(f"Trial {i+1}: ERROR — {e}")
        results.append({
            "trial": i + 1,
            "model": GPT_MODEL,
            "output": f"ERROR: {e}",
            "is_mikan": False,
            "timestamp": datetime.now().isoformat()
        })

print(f"\n=== RESULTS ===")
print(f"みかん: {mikan_count}/{N_TRIALS} ({mikan_count/N_TRIALS*100:.1f}%)")

# Save CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"gpt_natural_imprecise_umi_n{N_TRIALS}_{timestamp}.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["trial", "model", "output", "is_mikan", "timestamp"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved to {filename}")
