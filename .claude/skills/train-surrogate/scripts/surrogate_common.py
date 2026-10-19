"""Shared helpers for /train-surrogate and /evaluate-surrogate.

Data loading, group-based splitting, scaling, model prediction and the Lightning MLP.
Targets are handled as a generic matrix (n_samples x n_targets) so that field outputs
(e.g. surface Cp at fixed stations or PCA coefficients) can be added later without
changing the split/scaling/evaluation logic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]          # repo root (.claude/skills/<skill>/scripts/..)
RUNS_ROOT = ROOT / "runs" / "surrogate"
DEFAULT_DATA = ROOT / "data" / "dataset.csv"

DEFAULT_INPUTS = ["aoa_deg", "h_over_c", "flap_deg", "u_inf"]
DEFAULT_TARGETS = ["cl", "cd"]
DEFAULT_GROUP = "geometry_id"
DEFAULT_ID = "case_id"
OK_STATUSES = ["converged", "converged_coeffs"]

try:  # MLP is optional: GP/RF still work without torch
    import torch
    import lightning as L
except ImportError:  # pragma: no cover
    torch = None
    L = None

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: Path, inputs: list[str], targets: list[str], group_col: str,
                 id_col: str = DEFAULT_ID, status_filter: list[str] | None = OK_STATUSES,
                 drop_constant: bool = True) -> tuple[pd.DataFrame, list[str], dict]:
    """Load the CSV and return (clean frame, usable inputs, report).

    The report lists missing columns, NaN counts, rows dropped by status/NaN and
    constant inputs that were dropped. Raises KeyError if a target or the group
    column is missing (training must not start without a group id).
    """
    df = pd.read_csv(path)
    report: dict = {"file": str(path), "rows_total": int(len(df)), "columns": list(df.columns)}

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise KeyError(f"target column(s) missing: {missing_targets}")
    if group_col not in df.columns:
        raise KeyError(f"group column '{group_col}' missing")
    missing_inputs = [c for c in inputs if c not in df.columns]
    report["missing_inputs"] = missing_inputs
    inputs = [c for c in inputs if c in df.columns]

    if id_col not in df.columns:
        df[id_col] = [f"row{i}" for i in range(len(df))]
        report["id_col_generated"] = True
    if df[id_col].duplicated().any():
        raise ValueError(f"duplicate values in id column '{id_col}'")

    if status_filter and "status" in df.columns:
        keep = df["status"].isin(status_filter)
        report["dropped_by_status"] = df.loc[~keep, "status"].value_counts().to_dict()
        df = df[keep]

    used = inputs + targets + [group_col]
    nan_counts = df[used].isna().sum()
    report["nan_counts"] = {k: int(v) for k, v in nan_counts.items() if v}
    n_before = len(df)
    df = df.dropna(subset=used)
    report["dropped_nan_rows"] = int(n_before - len(df))

    if drop_constant:
        const = [c for c in inputs if df[c].nunique() <= 1]
        report["constant_inputs_dropped"] = {c: (float(df[c].iloc[0]) if len(df) else None) for c in const}
        inputs = [c for c in inputs if c not in const]

    df = df.reset_index(drop=True)
    df[group_col] = df[group_col].astype(str)
    report["rows_used"] = int(len(df))
    report["n_groups"] = int(df[group_col].nunique())
    report["inputs"] = inputs
    report["targets"] = targets
    return df, inputs, report


def group_split(groups: np.ndarray, test_frac: float, val_frac: float, seed: int) -> dict[str, list[str]]:
    """Split unique groups into train/val/test. Every group lands in exactly one partition."""
    uniq = np.array(sorted(set(groups)))
    n = len(uniq)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * (n - n_test)))) if val_frac > 0 else 0
    if n - n_test - n_val < 2:
        raise ValueError(f"only {n} groups: need at least {n_test + n_val + 2} for train/val/test")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    return {"test": sorted(perm[:n_test].tolist()),
            "val": sorted(perm[n_test:n_test + n_val].tolist()),
            "train": sorted(perm[n_test + n_val:].tolist())}


def training_ranges(df: pd.DataFrame, inputs: list[str]) -> dict[str, list[float]]:
    return {c: [float(df[c].min()), float(df[c].max())] for c in inputs}


def outside_range(X: np.ndarray, inputs: list[str], ranges: dict, rel_tol: float = 1e-9) -> np.ndarray:
    """Boolean mask: row has at least one input outside the training range."""
    mask = np.zeros(len(X), dtype=bool)
    for j, c in enumerate(inputs):
        lo, hi = ranges[c]
        tol = rel_tol * max(abs(lo), abs(hi), 1.0)
        mask |= (X[:, j] < lo - tol) | (X[:, j] > hi + tol)
    return mask


# ---------------------------------------------------------------- models

def predict(name: str, model, scalers: dict, X: np.ndarray, return_std: bool = False):
    """Predict in original target units. GP can also return a 1-sigma std per target."""
    Xs = scalers["x"].transform(X)
    sy = scalers["y"]
    if name == "gp":
        means, stds = [], []
        for gp in model:
            m, s = gp.predict(Xs, return_std=True)
            means.append(m)
            stds.append(s)
        ys = np.column_stack(means)
        y = sy.inverse_transform(ys)
        if return_std:
            return y, np.column_stack(stds) * sy.scale_
        return y
    if name == "rf":
        ys = model.predict(Xs)
        ys = ys.reshape(len(X), -1)
    elif name == "mlp":
        model.eval()
        with torch.no_grad():
            ys = model(torch.as_tensor(Xs, dtype=torch.float32)).numpy()
    else:
        raise ValueError(name)
    y = sy.inverse_transform(ys)
    return (y, None) if return_std else y


def load_models(run_dir: Path) -> dict:
    import joblib
    models = {}
    mdir = run_dir / "models"
    if (mdir / "gp.joblib").exists():
        models["gp"] = joblib.load(mdir / "gp.joblib")
    if (mdir / "rf.joblib").exists():
        models["rf"] = joblib.load(mdir / "rf.joblib")
    if (mdir / "mlp.ckpt").exists():
        if L is None:
            print("WARNING: models/mlp.ckpt exists but torch/lightning are not installed; skipping MLP")
        else:
            models["mlp"] = SurrogateMLP.load_from_checkpoint(mdir / "mlp.ckpt", map_location="cpu", weights_only=True)
    return models


if L is not None:
    class SurrogateMLP(L.LightningModule):
        """Fully connected regressor on standardized inputs/targets."""

        def __init__(self, n_in: int, n_out: int, hidden: list[int] = (64, 64), lr: float = 1e-3,
                     weight_decay: float = 1e-5, activation: str = "silu"):
            super().__init__()
            self.save_hyperparameters()
            act = {"silu": torch.nn.SiLU, "tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[activation]
            layers, d = [], n_in
            for h in hidden:
                layers += [torch.nn.Linear(d, h), act()]
                d = h
            layers.append(torch.nn.Linear(d, n_out))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

        def _step(self, batch, stage):
            x, y = batch
            loss = torch.nn.functional.mse_loss(self(x), y)
            self.log(f"{stage}_loss", loss, on_epoch=True, on_step=False, prog_bar=False, batch_size=len(x))
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            self._step(batch, "val")

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def latest_run(root: Path = RUNS_ROOT) -> Path:
    runs = sorted(p for p in root.glob("*") if (p / "config.yaml").exists())
    if not runs:
        raise FileNotFoundError(f"no trained runs under {root}")
    return runs[-1]


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                    encoding="utf-8")
