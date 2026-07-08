# GRPO Math Reasoning — Fine-tuning Qwen2.5-3B on GSM8K

Reinforcement-learning based fine-tuning of **Qwen2.5-3B-Instruct** to improve
step-by-step mathematical reasoning, using **GRPO (Group Relative Policy
Optimization)** — the same class of RL algorithm used behind reasoning models
like DeepSeek-R1. Trained end-to-end on a single Kaggle T4 GPU using LoRA and
4-bit quantization.

## Problem Statement

Small LLMs often produce correct-looking reasoning but arrive at wrong final
answers on multi-step math problems. This project explores whether **RL-based
fine-tuning (GRPO)**, rather than plain supervised fine-tuning, can push a 3B
model to reason more reliably on grade-school math (GSM8K).

## Approach

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Algorithm:** GRPO (via TRL's `GRPOTrainer`) — samples K rollouts per
  prompt, normalizes rewards within the group, and updates the policy without
  needing a separate value/critic model.
- **Efficient training:** 4-bit NF4 quantization (bitsandbytes) + LoRA
  (rank=16, alpha=32) on attention projections — fits training on a single T4
  16GB GPU.
- **Output format enforcement:** Model is prompted to reason inside
  `<think>...</think>` tags and give the final answer inside
  `<answer>...</answer>` tags.
- **Reward shaping:** Custom reward function scores completions on
  presence of the required tags, normalized per rollout group (GRPO-style
  advantage estimation).
- **Dataset:** [GSM8K](https://github.com/openai/grade-school-math) — grade
  school math word problems.

## Hyperparameters

| Parameter | Value |
|---|---|
| Rollouts per prompt (K) | 4 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-5 |
| KL penalty (beta) | 0.02 |
| Max new tokens | 256 |
| Training steps | 200 |

## Results

Evaluated on 50 unseen GSM8K test questions, using greedy decoding:

| Metric | Score |
|---|---|
| Accuracy (exact match on final numeric answer) | **52%** |
| Avg. reasoning trace length | ~105 words |

Training reward curves (format-adherence reward, KL divergence, entropy) are
logged and plotted from `trainer_state.json`.

## Limitations & Future Work

- Current reward function only checks **output format** (presence of
  `<think>`/`<answer>` tags), not mathematical correctness. Future work:
  add a correctness-based reward term (e.g., match final answer against
  ground truth) to more directly optimize accuracy rather than formatting.
- Evaluated on a limited sample (50 questions) due to Kaggle GPU time
  constraints; a full GSM8K test-set evaluation (1,319 questions) would give
  a more robust estimate.
- No baseline (pre-RL) accuracy is currently reported — adding a zero-shot
  base-model benchmark would make the improvement from GRPO training
  explicit and quantifiable.

## Tech Stack

`transformers` · `trl` (GRPOTrainer) · `peft` (LoRA) · `bitsandbytes`
(4-bit quantization) · `datasets` · PyTorch

## Repository Structure
# GRPO Math Reasoning — Fine-tuning Qwen2.5-3B on GSM8K

Reinforcement-learning based fine-tuning of **Qwen2.5-3B-Instruct** to improve
step-by-step mathematical reasoning, using **GRPO (Group Relative Policy
Optimization)** — the same class of RL algorithm used behind reasoning models
like DeepSeek-R1. Trained end-to-end on a single Kaggle T4 GPU using LoRA and
4-bit quantization.

## Problem Statement

Small LLMs often produce correct-looking reasoning but arrive at wrong final
answers on multi-step math problems. This project explores whether **RL-based
fine-tuning (GRPO)**, rather than plain supervised fine-tuning, can push a 3B
model to reason more reliably on grade-school math (GSM8K).

## Approach

- **Base model:** `Qwen/Qwen2.5-3B-Instruct`
- **Algorithm:** GRPO (via TRL's `GRPOTrainer`) — samples K rollouts per
  prompt, normalizes rewards within the group, and updates the policy without
  needing a separate value/critic model.
- **Efficient training:** 4-bit NF4 quantization (bitsandbytes) + LoRA
  (rank=16, alpha=32) on attention projections — fits training on a single T4
  16GB GPU.
- **Output format enforcement:** Model is prompted to reason inside
  `<think>...</think>` tags and give the final answer inside
  `<answer>...</answer>` tags.
- **Reward shaping:** Custom reward function scores completions on
  presence of the required tags, normalized per rollout group (GRPO-style
  advantage estimation).
- **Dataset:** [GSM8K](https://github.com/openai/grade-school-math) — grade
  school math word problems.

## Hyperparameters

| Parameter | Value |
|---|---|
| Rollouts per prompt (K) | 4 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Learning rate | 2e-5 |
| KL penalty (beta) | 0.02 |
| Max new tokens | 256 |
| Training steps | 200 |

## Results

Evaluated on 50 unseen GSM8K test questions, using greedy decoding:

| Metric | Score |
|---|---|
| Accuracy (exact match on final numeric answer) | **52%** |
| Avg. reasoning trace length | ~105 words |

Training reward curves (format-adherence reward, KL divergence, entropy) are
logged and plotted from `trainer_state.json`.

## Limitations & Future Work

- Current reward function only checks **output format** (presence of
  `<think>`/`<answer>` tags), not mathematical correctness. Future work:
  add a correctness-based reward term (e.g., match final answer against
  ground truth) to more directly optimize accuracy rather than formatting.
- Evaluated on a limited sample (50 questions) due to Kaggle GPU time
  constraints; a full GSM8K test-set evaluation (1,319 questions) would give
  a more robust estimate.
- No baseline (pre-RL) accuracy is currently reported — adding a zero-shot
  base-model benchmark would make the improvement from GRPO training
  explicit and quantifiable.

## Tech Stack

`transformers` · `trl` (GRPOTrainer) · `peft` (LoRA) · `bitsandbytes`
(4-bit quantization) · `datasets` · PyTorch

## Repository Structure

```text
├── README.md               # Project overview and benchmark results
├── train_grpo.py           # GRPO training script using GRPOTrainer and LoRA
├── evaluate_gsm8k.py       # Custom flexible parsing evaluation script
├── logs/
│   └── trainer_state.json  # Training logs containing rewards, step data, and KL metrics
└── assets/
    └── reward_curve.png    # Plotted reward standard deviation & tracking metrics
