# The Mikan Phenomenon

**Mora-level distributional attractors override explicit rules in large language models.**

This repository contains experiments, data, and paper source for the study of the *Mikan Phenomenon*: a systematic failure mode in which LLMs playing the Japanese word-chain game shiritori (しりとり) reach for みかん (mikan, tangerine) when the required response must begin with the mora /mi/—despite みかん ending in ん, which constitutes an immediate loss under the game's rules.

## What is the Mikan Phenomenon?

Shiritori (しりとり) is a Japanese word-chain game with two rules:
1. Each word must begin with the final mora of the previous word
2. Any word ending in ん results in immediate loss

Given any /mi/-ending word (うみ, かみ, しみ, etc.), frontier LLMs consistently produce みかん—demonstrating that distributional gravity toward high-frequency tokens can override explicitly represented constraints.

## Key Findings

- **Claude Sonnet 4.6**: 100% みかん across all conditions (fixed prompt n=50, natural precise rule n=50)
- **GPT-5.3 UI vs API discrepancy**: 3/3 みかん in ChatGPT UI, 0/50 in API—suggesting inference-time reasoning configuration differs between endpoints
- **Rule-precision paradox**: Precise rule statements → 100% みかん; imprecise statements → 64% みかん (Fisher's exact p=1.18×10⁻⁶)
- **Rule-cascade effect**: Non-みかん responses under imprecise conditions were predominantly りんご (11/50) and りす (9/50)—neither み-initial, indicating Rule 1 collapse
- **First-token commitment**: 29/50 correction attempts began with みか- prefix, indicating prefix-level commitment before constraint-checking
- **Cross-mora generalization**: /ki/ consistently produced きりん/きつね (5/5, both ん-terminal)
- **Longitudinal stability**: Claude みかん attractor stable across October 2025 → April 2026

## Repository Structure

```
mikan-phenomenon/
├── README.md
├── experiments/
│   ├── api/                          # Claude API experiments
│   │   ├── claude_fixed_prompt.py
│   │   ├── claude_natural_precise.py
│   │   └── claude_natural_imprecise.py
│   └── open_weight/                  # Open-weight model pilots
│       ├── shiritori_pilot.py        # llm-jp-3 pilot (v1)
│       ├── shiritori_pilot_v2.py     # llm-jp-3 pilot (v2, precise condition)
│       └── shiritori_qwen.py         # Qwen2.5-7B-Instruct pilot
├── data/
│   ├── claude_fixed_prompt_kami_n50_20260420.csv
│   ├── claude_natural_precise_umi_n50_20260420.csv
│   └── claude_natural_imprecise_umi_n50_20260420.csv
└── paper/
    ├── main.tex
    ├── references.bib
    └── neurips_2026.sty
```

## Data

All CSV files contain the following columns:
- `trial`: Trial number
- `output`: Full model output
- `is_mikan`: Boolean, True if output contains みかん
- `timestamp`: ISO 8601 timestamp (JST, UTC+9)

> **Note**: Timestamps reflect JST (UTC+9). Experiments were conducted on April 20, 2026 (US time).

## Reproducing API Experiments

```python
pip install anthropic
python experiments/api/claude_fixed_prompt.py
```

Requires `ANTHROPIC_API_KEY` environment variable.

## Reproducing Open-Weight Experiments

```python
pip install torch transformers accelerate bitsandbytes sentencepiece
python experiments/open_weight/shiritori_qwen.py
```

Tested on RTX 2080 Ti (11GB VRAM), Windows 11, CUDA 12.x, Python 3.11.

## Paper

Source in `paper/`. Compiled with pdflatex + NeurIPS 2026 style.

## Notes on Methodology

- Manual experiments used fresh private/incognito sessions per trial to eliminate conversational carryover
- API experiments used default sampling parameters (temperature=1.0)
- Model names reflect interface labels at time of testing
