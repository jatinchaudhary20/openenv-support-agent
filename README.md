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

## Overview
This project simulates a real-world customer support ticket resolution system.

## Features
- OpenEnv compatible environment
- 3 tasks: easy, medium, hard
- LLM-based agent (Hugging Face)
- Reward shaping with partial scoring
- Docker + HF deployment ready

## Actions
- classify()
- respond()
- resolve()
- escalate()

## Setup

```bash
pip install -r requirements.txt