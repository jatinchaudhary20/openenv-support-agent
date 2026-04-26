---
title: OpenEnv Support Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# OpenEnv Support Agent

## Hugging Face Space
- **Space URL:** <https://huggingface.co/spaces/chaudharyjatin20/openenv-support-agent>  

## Colab

- **Open in Colab:** <https://colab.research.google.com/drive/1aqo2huXfhNO_6fLBkrHRxwaAcP0knCmo?usp=sharing>  
- **Source file in repo:** [`colab_training.ipynb`](colab_training.ipynb)

## Writeup

- **Blog:** [`BLOG.md`](BLOG.md)
- **Blog URL:** *https://github.com/jatinchaudhary20/openenv-support-agent/blob/main/BLOG.md* 

## Training evidence (plots)

The repo includes **at least** a **loss** curve and a **mean reward** (rollout) series as **committed files** (not WandB-only, not only inside a Colab output cell):

| File | What it is |
|------|------------|
| [`artifacts/sft_loss_curve.png`](artifacts/sft_loss_curve.png) | SFT / training loss (example series; re-export after your run) |
| [`artifacts/env_reward_curve.png`](artifacts/env_reward_curve.png) | Environment reward proxy over steps (example series) |
| [`artifacts/training_loss_and_reward.png`](artifacts/training_loss_and_reward.png) | Combined figure (loss + reward, two panels) |
| [`artifacts/training_plots.png`](artifacts/training_plots.png) | Same data as a **single** chart (both lines, for slides) |

**Regenerate (optional):** `python scripts/render_submission_plots.py` (requires `pip install pillow` or install from `requirements.txt`).

**Training logs (JSON):** [`training_logs/training_metrics.json`](training_logs/training_metrics.json) (SFT `log_history` + optional `offline_eval`) and [`training_logs/env_eval.json`](training_logs/env_eval.json) (per-task reward). Both are **written** by `colab_training.ipynb` (sections 4 and 5); the repo also carries **starter** files you can replace after a full run. See [`training_logs/README.md`](training_logs/README.md). Hugging Face `outputs/` is gitignored.

### Inline plots (for judges)

![SFT loss curve](artifacts/sft_loss_curve.png)

![Environment reward curve](artifacts/env_reward_curve.png)

**Combined:** ![Training loss and reward](artifacts/training_loss_and_reward.png)

## The problem (story)

**Happy customers are not optional**—in a competitive market, retention and trust come from how support actually behaves, not from a single correct sentence.

Support is not one answer. It is a **sequence of decisions**: triage the issue, set priority, reply with empathy, optionally gather more feedback, and only then close the ticket. That workflow sounds simple, but in practice a good outcome depends on **order**, **tone**, and **reading the person**, not on ticking four boxes.

This project turns that idea into an **actionable environment**: the agent must **call tools in a sensible order**, and the reward signal can reflect **partial progress**, not only a final “done.” The design also makes room to reason about **user emotion**—because empathy and escalation paths matter as much as “correct” categories.

## What this environment is

- **OpenEnv-compatible**: HTTP server + tools (`MCPToolClient`) for reset, step, and rollouts.
- **Three tasks** (`easy`, `medium`, `hard`): different ticket themes and difficulty.
- **Structured state** (`ticket_state`, conversation, events) so you can **see** what went wrong or right.
- **Reward shaping** plus an optional **LLM judge** path for response quality (for example, empathy), so behavior is scored beyond a single yes or no.
- **Training path**: [colab_training.ipynb](colab_training.ipynb) shows **SFT** on conversational traces (Unsloth/TRL) and then **plugs the same client API** into evaluation (optional local server and an `evaluate_agent`-style pattern).

## Inspiration (why emotions)

The goal was a setting that is **industry-relevant** and still **pushes the agent** beyond a toy chat. Emotion-aware handling and realistic escalation pressure came from a simple “what if?”: what if the environment had to model **how the user feels**, not only what they typed? That nudge (including feedback from a welcome session to **try something a bit messier and more human**) shaped the focus on **tone, follow-up, and resolution quality**, not just classification accuracy.

## Why this is a strong evaluation (innovation)

- It tests a **multi-step** policy: classify → respond → (feedback) → resolve, not one-shot Q&A.
- It exposes **tool misuse and ordering**: resolving too early, skipping triage, or empty replies can cost you, depending on the reward design.
- It yields **trajectories** you can show in a demo: not only a single scalar at the end.

## Actions (MCP tools)

| Tool | Role |
|------|------|
| `classify` | Assign `category` and `priority` to the ticket. |
| `respond` | Send a first-line message to the user. |
| `feedback` | Model an extra user turn when your flow needs it. |
| `resolve` | Mark the ticket handled when the workflow allows it. |

(Aligned with `openenv.yaml` and `env/support_env.py`.)

## Rewards and training pipeline

- **Environment rewards** are defined in `SupportEnv` (structured ticket state, optional LLM judge).
- **Colab training** is **supervised (SFT)** on target traces: training **loss** tracks the **dataset**, while **behavior on the live environment** should be **reported separately** (for example, pre- and post-training eval on the same tasks and seeds, compared to a **baseline** such as a small rule policy).

**For rubrics that ask for “improvement in rewards”:** use **before/after** numbers (for example, mean return on `easy` / `medium` / `hard`) and/or **checkpoint** evals. A flat “reward” line on the same plot as loss, by itself, is **not** proof that the policy improved online.


## Project layout

- `env/` — `SupportEnv`, tickets, LLM judge helpers.  
- `server/app.py` — HTTP app, `/healthz`.  
- `openenv.yaml` — tasks, actions, and observation hints for tooling.  
