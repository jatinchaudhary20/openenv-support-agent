---
title: OpenEnv Support Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---
# OpenEnv Support Agent

## Problem Motivation
Support teams receive diverse tickets (billing, refund, technical) that require correct classification, empathetic communication, and consistent resolution steps. This project provides an OpenEnv environment to train and evaluate agent behavior on those tasks.

## Environment Design
The environment exposes three tools:
- `classify(category, priority)`
- `respond(message)`
- `resolve()`

Three task levels are supported:
- `easy`: billing issue
- `medium`: refund issue
- `hard`: technical issue

Rewards are hybrid and step-based:
- classification reward
- priority reward
- empathy reward
- resolution/penalty signals

## Project Structure
- `env/support_env.py`: environment state and reward logic
- `server/app.py`: OpenEnv HTTP server
- `inference.py`: model/fallback inference loop with logs
- `streamlit_app.py`: minimal UI demo
- `colab_training.ipynb`: SFT training + evaluation notebook

## Local Setup
```bash
/opt/anaconda3/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e .
```

## Run Backend
```bash
source .venv/bin/activate
python -m server.app
```

## Run Streamlit Demo
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Run Baseline Inference
```bash
source .venv/bin/activate
python inference.py
```

## Online Training (Environment-Connected)
```bash
source .venv/bin/activate
python train_online.py
```

This training loop interacts with the live environment on every step via `MCPToolClient`, trains a Q-learning policy for 180 episodes, and evaluates against a random baseline for 60 episodes.

Generated artifacts:
- `artifacts/training_reward_curve.png`
- `artifacts/baseline_vs_trained.png`
- `artifacts/training_metrics.json`

## Training Evidence (Latest Run)
- **Random baseline**
  - avg reward: `-1.3617`
  - resolution rate: `0.1833`
  - avg steps: `4.8333`
- **Trained policy**
  - avg reward: `5.7600`
  - resolution rate: `1.0000`
  - avg steps: `3.0000`

This demonstrates improvement from untrained/random behavior to a policy that reliably resolves tickets faster with higher reward.

## Results
- Multi-turn support flow is implemented.
- Hybrid reward shaping is implemented.
- HF model-first actioning with rule fallback is implemented.
- Structured per-step logs are emitted for evaluation.
- Environment-connected training + baseline comparison is implemented with quantitative results.

## Submission Links
- Hugging Face Space: `ADD_HF_SPACE_URL`
- Mini-blog / Video / Slides (writeup): `ADD_ARTIFACT_URL`
- Additional references: `ADD_OTHER_URLS`

Replace placeholders before final submission.