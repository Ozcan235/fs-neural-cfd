---
name: train-surrogate
description: Trains the OpenCFD surrogate models that predict front-wing aerodynamic coefficients (Cl, Cd) from CFD dataset inputs (angle of attack, ride height h/c, flap angle). Fits Gaussian Process and Random Forest baselines (scikit-learn) and an MLP (PyTorch Lightning) with a strict geometry-grouped train/val/test split, train-only scalers, fixed seeds, and saves everything under runs/surrogate/<timestamp>/. Use when the user asks to train, retrain, or fit the surrogate / ML model on data/dataset.csv or another CFD results CSV.
argument-hint: "[data=data/dataset.csv] [targets=cl,cd] [models=gp,rf,mlp] [seed=42] [tag=name]"
---

# /train-surrogate

Arguments: `$ARGUMENTS`

Trains surrogate models on CFD results. Current scope is **scalar coefficients** (decision (c):
scalars first, field data later). The scripts treat targets as a generic matrix, so field outputs
(e.g. surface Cp at fixed stations, or PCA coefficients of a field) can be added later as extra
target columns without changing the split/scaling logic.

## Steps

1. **Inspect the dataset first** (default `data/dataset.csv`, produced by `/sweep` and `/extract-coeffs`).
   Read the header and a few rows and tell the user, before training:
   - inputs: `aoa_deg`, `h_over_c`, `flap_deg`, `u_inf` (a constant input such as `u_inf` at fixed
     speed is dropped automatically and reported)
   - targets: `cl`, `cd` (Cl < 0 means downforce; the sign is kept)
   - group column: `geometry_id` (= h/c + flap configuration; AoA varies inside a group)
   - rows per `status`; only `converged` / `converged_coeffs` rows are used by default
   If the file does not exist, stop and say that `/sweep` has to produce data first.

2. **Group column gate.** If `geometry_id` (or the column the user named) is missing, **do not
   train**: ask the user which column identifies the geometry/configuration. If every group has
   only one row (e.g. data from a `/sweep` without `--aoa-values`), warn that the grouped split is
   effectively a per-sample split and ask whether to continue.

3. **Run training** from the repo root (it can take minutes because of GP restarts and MLP early
   stopping, so use `run_in_background: true` for datasets above ~300 rows):
   ```
   .venv/Scripts/python.exe .claude/skills/train-surrogate/scripts/train.py [--data data/dataset.csv] [--targets cl,cd] [--models gp,rf,mlp] [--seed 42] [--tag <name>]
   ```
   Useful options: `--inputs`, `--group-col`, `--test-frac 0.2` / `--val-frac 0.2` (fractions of
   **groups**, not rows), `--cv-folds 5`, `--hidden 64,64`, `--activation silu|tanh|relu`, `--lr`,
   `--epochs`, `--patience`, `--rf-trees`, `--status ''` (keep all rows), `--out`.

4. **Exit codes**: `0` success, `2` needs a user decision (missing group column, too few groups) –
   relay the message and ask; `1` error.

5. **Report** to the user:
   - inputs/targets used, NaN report, dropped rows, train/val/test group counts and the test group ids
   - GroupKFold CV metrics of GP and RF (RMSE ± std, R²) and the MLP best validation loss
   - every `WARNING` line, in particular the small-data warning (< 200 rows: the MLP may not beat
     the GP/RF baselines; < 10 groups: test metrics are high-variance)
   - the run folder, and suggest `/evaluate-surrogate <run>` as the next step

## What the script guarantees

- **Split by group, never by sample**: whole `geometry_id` groups go to train, val or test; the
  script asserts the partitions are disjoint and stores them in `split.json`.
- **Scalers fitted on training groups only** (train + val, never test); inside GroupKFold CV
  they are refit per fold. Saved to `scalers.joblib`.
- **Models**: GP (one per target, `Constant*RBF(ARD)+White`, 5 optimizer restarts, gives 1σ
  uncertainty), RF (500 trees), MLP (Lightning, AdamW, MSE on standardized targets, early stopping
  on the validation groups, best checkpoint kept). GP and RF train on train+val groups; the MLP
  trains on train groups and early-stops on val groups.
- **Reproducibility**: `--seed` drives numpy, the group split, the models and `lightning.seed_everything`
  with `deterministic=True`.

## Run folder layout (`runs/surrogate/<YYYYmmdd-HHMMSS>[_tag]/`)

| File | Content |
|---|---|
| `config.yaml` | data path + sha256, inputs, targets, group column, split sizes, training ranges, hyperparameters, CV metrics, warnings, package versions |
| `split.json` | train/val/test group ids and row ids |
| `scalers.joblib` | `{"x": StandardScaler, "y": StandardScaler}` |
| `models/gp.joblib`, `models/rf.joblib`, `models/mlp.ckpt` | trained models |
| `train_log.txt`, `mlp_logs/metrics.csv` | console log and per-epoch MLP losses |

`runs/` is git-ignored. Use the `scikit-learn` and `pytorch-lightning` skills for deeper changes
to the models.
