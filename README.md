# 🚀 OpenEnv Support Agent 

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Ready-green)](#)

A state-of-the-art, dynamic **Reinforcement Learning Environment** designed for evaluating and fine-tuning Agentic Support Systems. This project moves beyond static NLP testing by implementing a fully autonomous *LLM-as-a-Judge* scoring system and a stochastic ticket generator, creating a volatile and highly reactive gym for AI agents.

## 🔥 Key Features

- **🧠 LLM-as-a-Judge**: Empathy and qualitative responses are evaluated natively by an integrated Hugging Face Judge (`HuggingFaceTB/SmolLM2-1.7B-Instruct`), completely replacing brittle semantic regex checks with continuous logic scoring.
- **🌪️ Dynamic Ticket Synthesis**: Replaces static "easy/medium/hard" test cases with a `TicketGenerator` that programmatically builds hundreds of unique contexts incorporating distinct products, variable error codes, and dynamic financial scenarios.
- **😡 Reactive Escalation Mechanics**: The environment is genuinely hostile. If the acting agent provides a response that the LLM Judge deems un-empathetic, the environment overrides the ticket state—mutating it to simulate a "furious" customer and spiking the priority level mid-episode.
- **📊 Real-time UI Dashboard**: Directly integrates an intuitive HTML dashboard, plotting live metrics, trajectory actions, and running mock "Trained vs Random" benchmark comparisons mapped to the actual backend outputs.
- **🛠️ SFT Pipeline Ready**: Includes `train_online.py`—a complete fine-tuning bootstrap script utilizing `peft` and `trl` to train new base models over your generated environment trajectories.

## ⚡ Environment Actions

The agent interacts with the environment via standard MCP action patterns:
- `classify(category, priority)`: Identify the technical nature and urgency of the problem.
- `respond(message)`: Direct user chat interaction (dynamically evaluated by the Judge).
- `feedback(message)`: Parses external customer prompt injection into the current episode state.
- `resolve()`: Finalizes the conversation, computing the composite reward.

## 🛠️ Setup Instructions

This repository is purely modern and uses a unified `pyproject.toml` specification.

```bash
# Clone the repository
git clone https://github.com/jatinchaudhary20/openenv-support-agent
cd openenv-support-agent

# Standard Hackathon Install (Runs API, Environment, and Inference)
pip install .

# Advanced Install (Includes torch, transformers, trl for offline training)
pip install ".[train]"
```

## 🎮 Usage

### 1. Boot the OpenEnv Backend & UI
Starts the main FastAPI server initializing the complex Model Context Protocol logic.
```bash
python -m server.app
```
👉 *Head to `http://localhost:7860` in your browser to interact with the Live Demo Environment & Dashboard!*

### 2. Evaluate the Baseline Autonomous Agent
Automatically spins up a resilient, retrying HuggingFace agent that attempts to solve your environment.
```bash
export HF_TOKEN="your_huggingface_token"
python inference.py
```
*(Check the generated `evaluation_logs.jsonl` to see how well it handled the Angry Customer mechanics!)*

### 3. Run the Supervised Finetuning (SFT) Loop
Generates a massive synthetic environment corpus and fine-tunes a base model.
```bash
python train_online.py
```