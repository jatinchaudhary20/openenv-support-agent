# Training logs (JSON)

| File | Produced by | Contents |
|------|----------------|----------|
| `training_metrics.json` | **Section 4** of [`colab_training.ipynb`](../colab_training.ipynb) right after `trainer.train()` | `exported_at_utc`, `log_history` (SFT loss per step), optional `train_output`. May also include **`offline_eval`** if you committed aggregate baseline vs trained stats before running SFT (the notebook **merges** that block so it is not wiped). |
| `env_eval.json` | **Section 5** (OpenEnv + `evaluate_agent`) | `mean_reward` and **`per_task_total_return`** for `easy` / `medium` / `hard`. |

If you have **not** run the full notebook yet, you can still commit `training_metrics.json` with an empty `log_history` and an `offline_eval` object from your other experiments, then re-run later—Colab will fill `log_history` and keep `offline_eval` if it is already on disk.

**Do not** commit large `outputs/` checkpoints unless you use Git LFS; that directory is gitignored at the repo root.

Regenerate: run the full Colab from the repo root (`%cd` into this project). `train_online.py` only points to the notebook.
