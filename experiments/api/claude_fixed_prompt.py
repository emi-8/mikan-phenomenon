# Mikan Attractor API Experiment
# Claude Sonnet 4.6 — Fixed Prompt Condition
# Tests whether Claude produces みかん when the prior word ends in /mi/.
# Uses the same fixed prompt as the manual pilot experiment (April 17, 2026).

# Install: pip install anthropic

import anthropic
import csv
import time
from datetime import datetime

# Colab environment:
# from google.colab import userdata
# client = anthropic.Anthropic(api_key=userdata.get('ANTHROPIC_API_KEY'))

# Local environment:
import os
client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

SYSTEM_PROMPT = """あなたはしりとりゲームのプレイヤーBです。ルール：前の言葉の最後の文字から始まる言葉を言う。「ん」で終わったら負け。単語のみを答えてください。説明不要。"""

USER_PROMPT = """しりとりを続けます。

A: りんご
B: ごま
A: まど
B: どうろ
A: ろうそく
B: くま
A: まくら
B: らっぱ
A: ぱんち
B: ちから
A: らくだ
B: だるま
A: まめ
B: めだか
A: かみ
B:"""

MODEL = "claude-sonnet-4-6"
N_TRIALS = 50

results = []
mikan_count = 0

print(f"Running {N_TRIALS} trials (Fixed Prompt, prior word: かみ)...\n")

for i in range(N_TRIALS):
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT}
            ]
        )
        output = response.content[0].text.strip()
        is_mikan = "みかん" in output
        if is_mikan:
            mikan_count += 1
        results.append({
            "trial": i + 1,
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
            "output": f"ERROR: {e}",
            "is_mikan": False,
            "timestamp": datetime.now().isoformat()
        })

print(f"\n=== RESULTS ===")
print(f"みかん: {mikan_count}/{N_TRIALS} ({mikan_count/N_TRIALS*100:.1f}%)")

# Save CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"claude_fixed_prompt_kami_n{N_TRIALS}_{timestamp}.csv"
with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["trial", "output", "is_mikan", "timestamp"])
    writer.writeheader()
    writer.writerows(results)

print(f"Saved to {filename}")
