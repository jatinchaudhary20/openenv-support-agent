# Training logs (committed JSON)

This folder is filled when you run [`colab_training.ipynb`](../colab_training.ipynb) from the project root (after `%cd` into the repo).

**Examples (schema only):** [`training_metrics.example.json`](training_metrics.example.json), [`env_eval.example.json`](env_eval.example.json). Real runs overwrite the non-`.example` names below.

| File | When it is written | Contents |
|------|--------------------|----------|
| `training_metrics.json` | Right after `trainer.train()` | UTC timestamp, `log_history` (loss, etc. per `logging_steps`), and Hugging Face `train_output` fields when available. |
| `env_eval.json` | Section 5 (eval + plots) | Mean rollout return and per-task totals from `evaluate_agent` (OpenEnv on port 7860). |

**Commit these files** after a real run if your rubric requires “training evidence” outside Colab. Do not commit huge checkpoints under `outputs/`; add that dir to `.gitignore` (see project root) unless you use Git LFS.

Regenerate: run the full notebook; there is no separate offline trainer in `train_online.py` (it points to the notebook).
