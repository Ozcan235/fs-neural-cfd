"""Train surrogate models (GP, RF baselines + Lightning MLP) with a group-based split.

Usage (from repo root):
    .venv/Scripts/python.exe .claude/skills/train-surrogate/scripts/train.py \
        [--data data/dataset.csv] [--inputs aoa_deg,h_over_c,flap_deg] [--targets cl,cd] \
        [--group-col geometry_id] [--models gp,rf,mlp] [--seed 42] [--out runs/surrogate]

Exit codes: 0 ok, 1 error, 2 needs a user decision (no group column / too few groups).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import surrogate_common as sc  # noqa: E402

SMALL_DATA_ROWS = 200
SMALL_DATA_GROUPS = 10


class Tee:
    """Mirror stdout into the run's train_log.txt."""

    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def metrics(y_true: np.ndarray, y_pred: np.ndarray, targets: list[str]) -> dict:
    out = {}
    for j, t in enumerate(targets):
        e = y_pred[:, j] - y_true[:, j]
        ss_tot = np.sum((y_true[:, j] - y_true[:, j].mean()) ** 2)
        out[t] = {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                  "r2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan")}
    return out


def fit_scalers(X, Y):
    from sklearn.preprocessing import StandardScaler
    return {"x": StandardScaler().fit(X), "y": StandardScaler().fit(Y)}


def fit_gp(Xs, Ys, seed):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    d = Xs.shape[1]
    gps = []
    for j in range(Ys.shape[1]):
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(d), (1e-2, 1e3))
                  + WhiteKernel(1e-4, (1e-9, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=5, random_state=seed)
        gps.append(gp.fit(Xs, Ys[:, j]))
    return gps


def fit_rf(Xs, Ys, seed, n_estimators):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(Xs, Ys)


def group_cv(X, Y, groups, targets, models, seed, n_splits, rf_trees) -> dict:
    """GroupKFold CV on the training partition (scalers refit inside every fold)."""
    from sklearn.model_selection import GroupKFold
    res = {m: [] for m in models}
    for tr, va in GroupKFold(n_splits=n_splits).split(X, Y, groups):
        sc_ = fit_scalers(X[tr], Y[tr])
        Xs_tr, Ys_tr = sc_["x"].transform(X[tr]), sc_["y"].transform(Y[tr])
        for m in models:
            model = fit_gp(Xs_tr, Ys_tr, seed) if m == "gp" else fit_rf(Xs_tr, Ys_tr, seed, rf_trees)
            res[m].append(metrics(Y[va], sc.predict(m, model, sc_, X[va]), targets))
    summary = {}
    for m, folds in res.items():
        summary[m] = {t: {k: {"mean": float(np.mean([f[t][k] for f in folds])),
                              "std": float(np.std([f[t][k] for f in folds]))}
                          for k in ("mae", "rmse", "r2")} for t in targets}
    return summary


def train_mlp(X_tr, Y_tr, X_va, Y_va, scalers, args, run_dir: Path):
    import logging
    import warnings

    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    # Small tabular data on CPU: worker/checkpoint-dir hints and the logger tip are noise here
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*exists and is not empty.*")
    warnings.filterwarnings("ignore", message=".*treespec.*")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset

    L.seed_everything(args.seed, workers=True)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    ds_tr = TensorDataset(t(scalers["x"].transform(X_tr)), t(scalers["y"].transform(Y_tr)))
    ds_va = TensorDataset(t(scalers["x"].transform(X_va)), t(scalers["y"].transform(Y_va)))
    g = torch.Generator().manual_seed(args.seed)
    dl_tr = DataLoader(ds_tr, batch_size=min(args.batch_size, len(ds_tr)), shuffle=True, generator=g, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=len(ds_va), num_workers=0)

    model = sc.SurrogateMLP(X_tr.shape[1], Y_tr.shape[1], hidden=args.hidden, lr=args.lr,
                            weight_decay=args.weight_decay, activation=args.activation)
    ckpt = ModelCheckpoint(dirpath=run_dir / "models", filename="mlp", monitor="val_loss", mode="min",
                           save_top_k=1, enable_version_counter=False)
    trainer = L.Trainer(max_epochs=args.epochs, accelerator="cpu", deterministic=True,
                        logger=CSVLogger(run_dir, name="mlp_logs", version=""),
                        callbacks=[EarlyStopping("val_loss", patience=args.patience, mode="min"), ckpt],
                        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, dl_tr, dl_va)
    best = sc.SurrogateMLP.load_from_checkpoint(ckpt.best_model_path, map_location="cpu", weights_only=True)
    return best, {"best_val_loss": float(ckpt.best_model_score), "epochs_run": int(trainer.current_epoch)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=sc.DEFAULT_DATA)
    ap.add_argument("--inputs", default=",".join(sc.DEFAULT_INPUTS),
                    help="comma list; constant columns (e.g. u_inf at fixed speed) are dropped automatically")
    ap.add_argument("--targets", default=",".join(sc.DEFAULT_TARGETS))
    ap.add_argument("--group-col", default=sc.DEFAULT_GROUP)
    ap.add_argument("--id-col", default=sc.DEFAULT_ID)
    ap.add_argument("--status", default=",".join(sc.OK_STATUSES), help="keep rows with these status values ('' = all)")
    ap.add_argument("--models", default="gp,rf,mlp")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of GROUPS held out for test")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of remaining GROUPS for MLP validation")
    ap.add_argument("--cv-folds", type=int, default=5, help="GroupKFold folds on training groups (GP/RF)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf-trees", type=int, default=500)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--activation", default="silu", choices=["silu", "tanh", "relu"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", type=Path, default=sc.RUNS_ROOT)
    ap.add_argument("--tag", default="", help="optional suffix for the run folder name")
    args = ap.parse_args(argv)
    args.hidden = [int(h) for h in args.hidden.split(",") if h]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    status = [s for s in args.status.split(",") if s] or None

    np.random.seed(args.seed)
    if not args.data.exists():
        print(f"ERROR: dataset not found: {args.data}", file=sys.stderr)
        return 1
    try:
        df, inputs, rep = sc.load_dataset(args.data, args.inputs.split(","), targets, args.group_col,
                                          args.id_col, status)
    except KeyError as e:
        print(f"NEEDS USER DECISION: {e}. Training needs a group (geometry/configuration) id so that the "
              f"same geometry never appears in both train and test. Ask the user which column to use.",
              file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if "mlp" in models and sc.L is None:
        print("WARNING: torch/lightning not installed -> MLP skipped")
        models.remove("mlp")

    try:
        split = sc.group_split(df[args.group_col].to_numpy(), args.test_frac, args.val_frac if "mlp" in models else 0,
                               args.seed)
    except ValueError as e:
        print(f"NEEDS USER DECISION: {e}", file=sys.stderr)
        return 2
    if "mlp" in models and not split["val"]:
        print("ERROR: the MLP needs validation groups for early stopping; use --val-frac > 0", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"_{args.tag}" if args.tag else "")
    run_dir = args.out / stamp
    (run_dir / "models").mkdir(parents=True, exist_ok=False)
    sys.stdout = Tee(run_dir / "train_log.txt")
    t0 = time.time()

    print(f"Run: {run_dir}")
    print(f"Data: {args.data}  rows total={rep['rows_total']} used={rep['rows_used']} groups={rep['n_groups']}")
    print(f"  inputs : {inputs}")
    print(f"  targets: {targets}")
    print(f"  group  : {args.group_col}")
    if rep["missing_inputs"]:
        print(f"  missing input columns (ignored): {rep['missing_inputs']}")
    if rep.get("constant_inputs_dropped"):
        print(f"  constant inputs dropped: {rep['constant_inputs_dropped']}")
    if rep.get("dropped_by_status"):
        print(f"  rows dropped by status: {rep['dropped_by_status']}")
    print(f"  NaN counts: {rep['nan_counts'] or 'none'}  (rows dropped: {rep['dropped_nan_rows']})")

    warnings = []
    if rep["rows_used"] < SMALL_DATA_ROWS:
        warnings.append(f"Only {rep['rows_used']} samples (< {SMALL_DATA_ROWS}): the MLP may not beat the GP/RF "
                        "baselines; prefer the GP for small CFD datasets.")
    if rep["n_groups"] < SMALL_DATA_GROUPS:
        warnings.append(f"Only {rep['n_groups']} geometry groups: the test set has very few groups, so test "
                        "metrics have high variance.")
    for w in warnings:
        print(f"WARNING: {w}")

    g = df[args.group_col]
    part = {k: df[g.isin(v)] for k, v in split.items()}
    assert not (set(split["train"]) | set(split["val"])) & set(split["test"])
    fit_df = df[~g.isin(split["test"])]                       # train + val groups
    print(f"Split by group: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])} groups "
          f"-> rows {len(part['train'])}/{len(part['val'])}/{len(part['test'])}")
    print(f"  test groups: {split['test']}")

    X_fit, Y_fit = fit_df[inputs].to_numpy(float), fit_df[targets].to_numpy(float)
    scalers = fit_scalers(X_fit, Y_fit)                       # fitted on training groups only
    import joblib
    joblib.dump(scalers, run_dir / "scalers.joblib")

    cv = {}
    base = [m for m in models if m in ("gp", "rf")]
    n_fit_groups = fit_df[args.group_col].nunique()
    folds = min(args.cv_folds, n_fit_groups)
    if base and folds >= 2:
        print(f"GroupKFold CV ({folds} folds) on training groups ...")
        cv = group_cv(X_fit, Y_fit, fit_df[args.group_col].to_numpy(), targets, base, args.seed, folds, args.rf_trees)
        for m, r in cv.items():
            print(f"  {m}: " + "  ".join(f"{t} RMSE={r[t]['rmse']['mean']:.4g}±{r[t]['rmse']['std']:.2g} "
                                         f"R2={r[t]['r2']['mean']:.3f}" for t in targets))

    Xs, Ys = scalers["x"].transform(X_fit), scalers["y"].transform(Y_fit)
    info = {}
    if "gp" in models:
        gp = fit_gp(Xs, Ys, args.seed)
        joblib.dump(gp, run_dir / "models" / "gp.joblib")
        info["gp"] = {"kernels": [str(m.kernel_) for m in gp]}
        print("GP trained. Kernels: " + " | ".join(info["gp"]["kernels"]))
    if "rf" in models:
        rf = fit_rf(Xs, Ys, args.seed, args.rf_trees)
        joblib.dump(rf, run_dir / "models" / "rf.joblib")
        info["rf"] = {"n_estimators": args.rf_trees}
        print("RF trained.")
    if "mlp" in models:
        print("MLP training (early stopping on validation groups) ...")
        _, info["mlp"] = train_mlp(part["train"][inputs].to_numpy(float), part["train"][targets].to_numpy(float),
                                   part["val"][inputs].to_numpy(float), part["val"][targets].to_numpy(float),
                                   scalers, args, run_dir)
        print(f"MLP trained: {info['mlp']}")

    import sklearn
    cfg = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "data": {"path": str(args.data.resolve()), "sha256": sc.sha256(args.data), "report": rep},
        "inputs": inputs, "targets": targets, "group_col": args.group_col, "id_col": args.id_col,
        "status_filter": status,
        "split": {"method": "random group split", "test_frac": args.test_frac, "val_frac": args.val_frac,
                  "n_groups": {k: len(v) for k, v in split.items()}},
        "scalers": {"type": "StandardScaler (x and y)", "fitted_on": "train+val groups (no test rows)",
                    "n_rows": int(len(fit_df))},
        "training_ranges": sc.training_ranges(fit_df, inputs),
        "models": models,
        "hyperparameters": {
            "gp": {"kernel": "Constant*RBF(ARD)+White", "n_restarts_optimizer": 5},
            "rf": {"n_estimators": args.rf_trees},
            "mlp": {"hidden": args.hidden, "activation": args.activation, "lr": args.lr,
                    "weight_decay": args.weight_decay, "epochs": args.epochs, "patience": args.patience,
                    "batch_size": args.batch_size},
        },
        "cv": cv, "fit_info": info, "warnings": warnings,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                     "torch": getattr(sc.torch, "__version__", None),
                     "lightning": getattr(sc.L, "__version__", None)},
    }
    import yaml
    (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), sort_keys=False),
                                         encoding="utf-8")
    sc.dump_json(run_dir / "split.json", {
        "group_col": args.group_col, "id_col": args.id_col,
        "train_groups": split["train"], "val_groups": split["val"], "test_groups": split["test"],
        **{f"{k}_ids": part[k][args.id_col].astype(str).tolist() for k in ("train", "val", "test")},
    })
    print(f"Done in {time.time() - t0:.1f} s. Saved: config.yaml, split.json, scalers.joblib, models/, train_log.txt")
    print(f"Next: /evaluate-surrogate {run_dir.name}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout
    return 0


if __name__ == "__main__":
    sys.exit(main())
