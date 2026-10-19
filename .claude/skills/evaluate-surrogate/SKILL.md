---
name: evaluate-surrogate
description: Evaluates a trained OpenCFD surrogate run (runs/surrogate/<run>/) only on its held-out, never-seen geometry groups. Verifies there is no data leakage (disjoint train/test geometry ids, train-only scalers) and stops if a check fails, compares GP, RF and MLP with MAE/RMSE/R²/percent error, checks physical consistency (negative Cd, Cl-vs-AoA shape, extrapolation), and writes metrics.json, figures and report.md with a limitations section. Use after /train-surrogate or when the user asks how good, accurate, or trustworthy the surrogate model is.
argument-hint: "[run=latest|<run-folder>] [data=<csv>]"
---

# /evaluate-surrogate

Arguments: `$ARGUMENTS`

## Steps

1. **Pick the run.** Default is the newest folder under `runs/surrogate/`. If the user names one,
   pass the folder name or path. If no run exists, say so and suggest `/train-surrogate`.

2. **Run the evaluation** from the repo root:
   ```
   .venv/Scripts/python.exe .claude/skills/evaluate-surrogate/scripts/evaluate.py [--run latest|<folder>] [--data <csv>]
   ```
   `--data` defaults to the CSV recorded in the run's `config.yaml`. If that file changed since
   training (sha256 differs) the script still evaluates **only the recorded test rows** and says so;
   mention this to the user.

3. **Leakage gate (exit code 2).** Before any metric is computed the script checks:
   - train ∩ test and val ∩ test geometry groups are empty, and train ∩ test row ids are empty
   - every test row in the current data still belongs to a test group (no relabelled `geometry_id`)
   - the x and y scalers were fitted on exactly the train+val rows (sample count and mean/std
     match the training rows and do **not** match train+test)
   If any check fails, **stop**: do not compute or quote metrics. Explain which check failed and why
   (details are in `evaluation/leakage_check.json`) and suggest retraining with `/train-surrogate`.

4. **Report the results** (exit code 0):
   - one comparison table for GP, RF and MLP: MAE, RMSE, R², mean and max percent error per target,
     and the best model (lowest normalized RMSE). Percent error excludes points with |CFD| ≈ 0.
   - physical consistency: negative Cd predictions (test rows and 2000 random in-range points),
     Cl-vs-AoA shape flags per test geometry (slope sign vs the CFD trend, wiggles, oscillation).
     Near stall a non-monotonic Cl(AoA) can be physical, so present flags as warnings.
   - extrapolation: test rows outside the training range get a **separate warning** and separate
     metrics; never mix them into the headline numbers without saying so.
   - a short reading of the figures (open the PNGs with Read and describe what they show)
   - the "Bu modelin sınırları" section of `report.md` in your own words: reliable input ranges,
     weakest region, worst cases, and how few test groups the numbers rest on.
   Be honest about uncertainty: with few test groups, differences between models of a few percent
   are not meaningful.

## Outputs (`runs/surrogate/<run>/evaluation/`)

| File | Content |
|---|---|
| `leakage_check.json` | every leakage check with pass/fail and details |
| `metrics.json` | per-model, per-target metrics; best model; extrapolation rows and their separate metrics |
| `physics_checks.json` | negative-Cd counts, Cl(AoA) slope/shape diagnostics per test geometry |
| `figures/parity_<target>.png` | predicted vs CFD with the y = x line, one panel per model |
| `figures/error_hist_<target>.png` | error distribution per model |
| `figures/error_vs_inputs_<target>.png` | error against each input parameter |
| `figures/cl_vs_aoa_test_geometries.png` | Cl(AoA) curves of every model (GP ±2σ band) against the CFD test points |
| `report.md` | summary report ending with "Bu modelin sınırları" (limitations of this model) |

## Notes

- Figures follow publication practice: colour-blind-safe Okabe-Ito colours, redundant markers and
  line styles, labelled axes with units, explicit y = x reference.
- Field outputs (decision (c), phase 2) can reuse the leakage gate and metric code: add field
  coefficients as extra targets; the per-target metrics and parity plots work unchanged.
