"""Evaluate a trained surrogate run on its held-out (never seen) geometry groups.

Usage (from repo root):
    .venv/Scripts/python.exe .claude/skills/evaluate-surrogate/scripts/evaluate.py [--run latest|<folder>|<path>] [--data <csv>]

Stops with exit code 2 if a leakage check fails. Writes runs/surrogate/<run>/evaluation/
(metrics.json, leakage_check.json, physics_checks.json, figures/*.png, report.md).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train-surrogate" / "scripts"))
import surrogate_common as sc  # noqa: E402

OKABE_ITO = {"gp": "#0072B2", "rf": "#E69F00", "mlp": "#009E73", "cfd": "#000000"}
MARKERS = {"gp": "o", "rf": "s", "mlp": "^"}
LABELS = {"gp": "GP", "rf": "Random Forest", "mlp": "MLP"}
UNITS = {"aoa_deg": "AoA [deg]", "h_over_c": "h/c [-]", "flap_deg": "flap angle [deg]", "u_inf": "U∞ [m/s]",
         "cl": "Cl [-]", "cd": "Cd [-]"}


def resolve_run(arg: str) -> Path:
    if arg in ("", "latest"):
        return sc.latest_run()
    p = Path(arg)
    if not p.is_absolute() and not p.exists():
        p = sc.RUNS_ROOT / arg
    if not (p / "config.yaml").exists():
        raise FileNotFoundError(f"{p} is not a training run (config.yaml missing)")
    return p.resolve()


# ---------------------------------------------------------------- leakage checks

def leakage_checks(df, cfg, split, scalers) -> list[dict]:
    gcol, icol = split["group_col"], split["id_col"]
    tr, va, te = set(split["train_groups"]), set(split["val_groups"]), set(split["test_groups"])
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    add("train ∩ test groups is empty", not (tr & te), sorted(tr & te) or "no shared group")
    add("val ∩ test groups is empty", not (va & te), sorted(va & te) or "no shared group")
    add("test set is not empty", len(te) > 0 and len(split["test_ids"]) > 0, f"{len(te)} groups")

    ids = df[icol].astype(str)
    fit_ids = set(split["train_ids"]) | set(split["val_ids"])
    add("train ∩ test row ids is empty", not (fit_ids & set(split["test_ids"])),
        sorted(fit_ids & set(split["test_ids"]))[:10] or "no shared row")
    missing = set(split["test_ids"]) - set(ids)
    add("all test rows still present in the data", not missing, sorted(missing)[:10] or "ok")

    # Group labels in the current data must agree with the recorded split
    row_group = dict(zip(ids, df[gcol].astype(str)))
    bad_test = [i for i in split["test_ids"] if i in row_group and row_group[i] not in te]
    bad_fit = [i for i in fit_ids if i in row_group and row_group[i] in te]
    add("test rows belong only to test groups (data re-check)", not bad_test and not bad_fit,
        {"test_rows_outside_test_groups": bad_test[:10], "train_rows_inside_test_groups": bad_fit[:10]}
        if bad_test or bad_fit else "ok")

    # Scaler must have been fitted on exactly the train+val rows
    fit_rows = df[ids.isin(fit_ids)]
    X_fit = fit_rows[cfg["inputs"]].to_numpy(float)
    Y_fit = fit_rows[cfg["targets"]].to_numpy(float)
    for key, A in (("x", X_fit), ("y", Y_fit)):
        s = scalers[key]
        same_n = int(s.n_samples_seen_) == len(A)
        same_stats = same_n and np.allclose(s.mean_, A.mean(0), rtol=1e-9, atol=1e-12) and \
            np.allclose(s.scale_, A.std(0), rtol=1e-9, atol=1e-12)
        all_rows = df[ids.isin(fit_ids | set(split["test_ids"]))]
        B = all_rows[cfg["inputs"] if key == "x" else cfg["targets"]].to_numpy(float)
        matches_all = int(s.n_samples_seen_) == len(B) and np.allclose(s.mean_, B.mean(0), rtol=1e-9)
        add(f"{key}-scaler fitted on training rows only", same_stats and not matches_all,
            {"n_seen": int(s.n_samples_seen_), "n_train_rows": len(A), "n_train_plus_test": len(B),
             "matches_train_stats": bool(same_stats), "matches_train_plus_test_stats": bool(matches_all)})
    return checks


# ---------------------------------------------------------------- metrics

def target_metrics(y, p) -> dict:
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    eps = 1e-3 * max(np.max(np.abs(y)), 1e-12)          # avoid exploding % error near y = 0
    ok = np.abs(y) > eps
    pct = 100 * np.abs(e[ok]) / np.abs(y[ok])
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "r2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else None,
            "mean_pct_error": float(pct.mean()) if len(pct) else None,
            "max_pct_error": float(pct.max()) if len(pct) else None,
            "pct_error_excluded_near_zero": int((~ok).sum())}


def all_metrics(Y, preds, targets, mask=None) -> dict:
    out = {}
    for m, P in preds.items():
        sel = slice(None) if mask is None else mask
        if mask is not None and not mask.any():
            continue
        out[m] = {t: target_metrics(Y[sel, j], P[sel, j]) for j, t in enumerate(targets)}
    return out


# ---------------------------------------------------------------- physics checks

def physics_checks(df_test, models, scalers, cfg, args, seed) -> dict:
    inputs, targets, gcol = cfg["inputs"], cfg["targets"], cfg["group_col"]
    rng = np.random.default_rng(seed)
    ranges = cfg["training_ranges"]
    res = {"cd_negative": {}, "cl_vs_aoa": {}, "notes": []}

    if args.cd_col in targets:
        jcd = targets.index(args.cd_col)
        Xbox = np.column_stack([rng.uniform(*ranges[c], 2000) for c in inputs])
        Xte = df_test[inputs].to_numpy(float)
        for m, model in models.items():
            p_te = sc.predict(m, model, scalers, Xte)[:, jcd]
            p_box = sc.predict(m, model, scalers, Xbox)[:, jcd]
            res["cd_negative"][m] = {"test_points": int((p_te < 0).sum()), "test_n": int(len(p_te)),
                                     "random_in_range_points": int((p_box < 0).sum()), "random_n": int(len(p_box)),
                                     "min_predicted_cd": float(min(p_te.min(), p_box.min()))}
    else:
        res["notes"].append(f"'{args.cd_col}' is not a target: Cd sign check skipped")

    if args.cl_col in targets and args.aoa_col in inputs:
        jcl, ja = targets.index(args.cl_col), inputs.index(args.aoa_col)
        a_lo, a_hi = ranges[args.aoa_col]
        grid = np.linspace(a_lo, a_hi, 61)
        for g, sub in df_test.groupby(gcol):
            base = sub[inputs].iloc[0].to_numpy(float)
            Xg = np.repeat(base[None, :], len(grid), 0)
            Xg[:, ja] = grid
            cfd_slope = (np.polyfit(sub[args.aoa_col], sub[args.cl_col], 1)[0]
                         if sub[args.aoa_col].nunique() >= 2 else None)
            entry = {"cfd_points": int(len(sub)), "cfd_slope_per_deg": cfd_slope, "models": {}}
            for m, model in models.items():
                cl = sc.predict(m, model, scalers, Xg)[:, jcl]
                d = np.diff(cl)
                slope = float(np.polyfit(grid, cl, 1)[0])
                sign_changes = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-9])) != 0))
                tv_ratio = float(np.sum(np.abs(d)) / max(abs(cl[-1] - cl[0]), 1e-12))
                flags = []
                if cfd_slope is not None and np.sign(slope) != np.sign(cfd_slope):
                    flags.append("overall slope sign differs from CFD trend")
                if sign_changes > 2:
                    flags.append(f"dCl/dAoA changes sign {sign_changes} times (wiggly)")
                if tv_ratio > 3:
                    flags.append(f"total variation {tv_ratio:.1f}x net change (oscillatory)")
                entry["models"][m] = {"slope_per_deg": slope, "derivative_sign_changes": sign_changes,
                                      "total_variation_ratio": tv_ratio, "flags": flags}
            res["cl_vs_aoa"][str(g)] = entry
        res["notes"].append("Non-monotonic Cl(AoA) can be physical near stall; flags are warnings, not failures.")
    else:
        res["notes"].append("Cl-vs-AoA shape check skipped (cl target or aoa input missing)")
    return res


# ---------------------------------------------------------------- figures

def make_figures(fig_dir, Y, preds, X, cfg, df_test, models, scalers, args) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3})
    inputs, targets = cfg["inputs"], cfg["targets"]
    names = list(preds)
    files = []

    for j, t in enumerate(targets):
        # Parity plots
        fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.2), layout="constrained", squeeze=False)
        lo, hi = min(Y[:, j].min(), *(P[:, j].min() for P in preds.values())), \
            max(Y[:, j].max(), *(P[:, j].max() for P in preds.values()))
        pad = 0.05 * (hi - lo or 1)
        for ax, m in zip(axes[0], names):
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
            ax.scatter(Y[:, j], preds[m][:, j], s=18, marker=MARKERS[m], color=OKABE_ITO[m], edgecolor="none",
                       label=LABELS[m])
            e = preds[m][:, j] - Y[:, j]
            r2 = 1 - np.sum(e ** 2) / max(np.sum((Y[:, j] - Y[:, j].mean()) ** 2), 1e-30)
            ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), aspect="equal",
                   xlabel=f"CFD {UNITS.get(t, t)}", ylabel=f"predicted {UNITS.get(t, t)}",
                   title=f"{LABELS[m]}  (R² = {r2:.3f}, n = {len(e)})")
            ax.legend(loc="upper left", fontsize=7)
        f = fig_dir / f"parity_{t}.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)

        # Error histograms
        fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")
        all_e = np.concatenate([P[:, j] - Y[:, j] for P in preds.values()])
        bins = np.linspace(all_e.min(), all_e.max(), 21) if np.ptp(all_e) > 0 else 10
        for m in names:
            ax.hist(preds[m][:, j] - Y[:, j], bins=bins, histtype="step", lw=1.5, color=OKABE_ITO[m],
                    label=LABELS[m], linestyle={"gp": "-", "rf": "--", "mlp": ":"}[m])
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel=f"prediction − CFD ({t})", ylabel="count (test rows)", title=f"Error distribution: {t}")
        ax.legend(fontsize=7)
        f = fig_dir / f"error_hist_{t}.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)

        # Error vs each input
        fig, axes = plt.subplots(1, len(inputs), figsize=(3.2 * len(inputs), 3), layout="constrained", squeeze=False)
        for ax, (k, c) in zip(axes[0], enumerate(inputs)):
            for m in names:
                ax.scatter(X[:, k], preds[m][:, j] - Y[:, j], s=16, marker=MARKERS[m], color=OKABE_ITO[m],
                           edgecolor="none", alpha=0.8, label=LABELS[m])
            ax.axhline(0, color="k", lw=0.8)
            ax.set(xlabel=UNITS.get(c, c), ylabel=f"error in {t}")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"Test error vs. input parameters: {t}")
        f = fig_dir / f"error_vs_inputs_{t}.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)

    # Cl vs AoA per test geometry
    if args.cl_col in targets and args.aoa_col in inputs:
        jcl, ja = targets.index(args.cl_col), inputs.index(args.aoa_col)
        groups = list(df_test.groupby(cfg["group_col"]))[:12]
        ncol = min(4, len(groups))
        nrow = int(np.ceil(len(groups) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), layout="constrained", squeeze=False)
        a_lo, a_hi = cfg["training_ranges"][args.aoa_col]
        grid = np.linspace(a_lo, a_hi, 61)
        for ax, (g, sub) in zip(axes.flat, groups):
            base = sub[inputs].iloc[0].to_numpy(float)
            Xg = np.repeat(base[None, :], len(grid), 0)
            Xg[:, ja] = grid
            for m, model in models.items():
                mean, std = sc.predict(m, model, scalers, Xg, return_std=True)
                ax.plot(grid, mean[:, jcl], color=OKABE_ITO[m], lw=1.4, label=LABELS[m],
                        linestyle={"gp": "-", "rf": "--", "mlp": ":"}[m])
                if std is not None:
                    ax.fill_between(grid, mean[:, jcl] - 2 * std[:, jcl], mean[:, jcl] + 2 * std[:, jcl],
                                    color=OKABE_ITO[m], alpha=0.15, lw=0, label="GP ±2σ")
            ax.scatter(sub[args.aoa_col], sub[args.cl_col], color=OKABE_ITO["cfd"], s=20, zorder=5, label="CFD (test)")
            ax.set(title=str(g), xlabel=UNITS.get(args.aoa_col), ylabel=UNITS.get(args.cl_col))
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle("Cl vs AoA on unseen test geometries")
        f = fig_dir / "cl_vs_aoa_test_geometries.png"
        fig.savefig(f)
        plt.close(fig)
        files.append(f.name)
    return files


# ---------------------------------------------------------------- report

def fmt(v, nd=4):
    return "–" if v is None else f"{v:.{nd}g}"


def weak_regions(X, Y, preds, cfg, best) -> list[str]:
    """Input tertiles with the highest normalized abs error for the best model."""
    lines = []
    inputs, targets = cfg["inputs"], cfg["targets"]
    err = np.mean([np.abs(preds[best][:, j] - Y[:, j]) / (Y[:, j].std() or 1) for j in range(len(targets))], axis=0)
    for k, c in enumerate(inputs):
        if np.unique(X[:, k]).size < 3:
            continue
        edges = np.quantile(X[:, k], [0, 1 / 3, 2 / 3, 1])
        idx = np.clip(np.digitize(X[:, k], edges[1:-1]), 0, 2)
        means = [err[idx == b].mean() if (idx == b).any() else np.nan for b in range(3)]
        b = int(np.nanargmax(means))
        lines.append(f"- `{c}` in [{edges[b]:.3g}, {edges[b + 1]:.3g}]: mean normalized error {means[b]:.3g} "
                     f"(other tertiles {', '.join(f'{x:.3g}' for i, x in enumerate(means) if i != b)})")
    return lines


def write_report(path, run_dir, cfg, checks, met, met_in, met_out, n_out, phys, figs, X, Y, preds, df_test, best):
    t = cfg["targets"]
    L_ = [f"# Surrogate evaluation: `{run_dir.name}`", "",
          f"Generated {datetime.now().isoformat(timespec='seconds')}. Test set = "
          f"{len(set(df_test[cfg['group_col']]))} unseen `{cfg['group_col']}` groups, {len(df_test)} rows. "
          f"Inputs: {', '.join(cfg['inputs'])}. Targets: {', '.join(t)}.", "",
          "## Leakage checks", "", "| Check | Result |", "|---|---|"]
    L_ += [f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} |" for c in checks]
    L_ += ["", "## Test metrics (unseen geometries)", "",
           "| Model | Target | MAE | RMSE | R² | mean % err | max % err |", "|---|---|---|---|---|---|---|"]
    for m, r in met.items():
        for tt in t:
            x = r[tt]
            L_.append(f"| {LABELS[m]}{' **(best)**' if m == best else ''} | {tt} | {fmt(x['mae'])} | {fmt(x['rmse'])} | "
                      f"{fmt(x['r2'], 3)} | {fmt(x['mean_pct_error'], 3)} | {fmt(x['max_pct_error'], 3)} |")
    L_ += ["", "Percent errors exclude points with |CFD| < 0.1 % of the largest |value| (near-zero Cl).", ""]
    if cfg.get("cv"):
        L_ += ["GroupKFold CV on training groups (RMSE mean ± std): " + "; ".join(
            f"{LABELS[m]} " + ", ".join(f"{tt} {v[tt]['rmse']['mean']:.3g}±{v[tt]['rmse']['std']:.2g}" for tt in t)
            for m, v in cfg["cv"].items()), ""]

    L_ += ["## Extrapolation", ""]
    if n_out:
        L_ += [f"**WARNING:** {n_out} test rows have inputs outside the training range; they are reported "
               "separately below and should not be trusted.", ""]
        for name, mm in (("inside range", met_in), ("outside range", met_out)):
            for m, r in mm.items():
                L_.append(f"- {name}, {LABELS[m]}: " + ", ".join(f"{tt} RMSE {r[tt]['rmse']:.3g}" for tt in t))
        L_.append("")
    else:
        L_ += ["All test inputs lie inside the training range (no extrapolation in this test set).", ""]

    L_ += ["## Physical consistency", ""]
    for m, r in phys["cd_negative"].items():
        L_.append(f"- {LABELS[m]}: negative Cd on {r['test_points']}/{r['test_n']} test rows and "
                  f"{r['random_in_range_points']}/{r['random_n']} random in-range points (min Cd {r['min_predicted_cd']:.3g})")
    flagged = [(g, m, f) for g, e in phys["cl_vs_aoa"].items() for m, v in e["models"].items() for f in v["flags"]]
    if phys["cl_vs_aoa"]:
        L_.append(f"- Cl(AoA) shape on {len(phys['cl_vs_aoa'])} test geometries: "
                  + ("no flags." if not flagged else f"{len(flagged)} flag(s):"))
        L_ += [f"  - `{g}` {LABELS[m]}: {f}" for g, m, f in flagged[:15]]
    L_ += [f"- {n}" for n in phys["notes"]]
    L_ += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figs]

    # Limitations
    rng = cfg["training_ranges"]
    rows = cfg["data"]["report"]["rows_used"]
    ngroups = cfg["data"]["report"]["n_groups"]
    order = np.argsort(-np.mean([np.abs(preds[best][:, j] - Y[:, j]) / (Y[:, j].std() or 1)
                                 for j in range(len(t))], axis=0))[:3]
    L_ += ["", "## Bu modelin sınırları (limitations of this model)", "",
           "**Reliable range** (training data; outside it every prediction is an extrapolation):", ""]
    L_ += [f"- `{c}`: {lo:.4g} … {hi:.4g}" for c, (lo, hi) in rng.items()]
    dropped = cfg["data"]["report"].get("constant_inputs_dropped") or {}
    if dropped:
        L_ += [f"- held constant during training (the model knows nothing about other values): "
               + ", ".join(f"`{c}` = {v}" for c, v in dropped.items())]
    L_ += ["", f"**Evidence base:** {rows} CFD samples in {ngroups} geometry groups; test metrics come from only "
           f"{len(set(df_test[cfg['group_col']]))} groups, so they carry substantial sampling uncertainty.", ""]
    L_ += [f"**Where `{LABELS[best]}` is weakest** (input tertiles with the largest normalized error):", ""]
    L_ += weak_regions(X, Y, preds, cfg, best) or ["- not enough distinct input values to localize errors"]
    L_ += ["", "Worst test cases:", ""]
    for i in order:
        row = df_test.iloc[int(i)]
        L_.append(f"- `{row[cfg['id_col']]}`: " + ", ".join(f"{c}={row[c]:.4g}" for c in cfg["inputs"]) + " → " +
                  ", ".join(f"{tt} CFD {Y[i, j]:.4g} vs {preds[best][i, j]:.4g}" for j, tt in enumerate(t)))
    L_ += ["", "**General caveats:**", "",
           "- The surrogate cannot be more accurate than the CFD it was trained on (2D RANS, kOmegaSST, isolated "
           "wing; see README scope).",
           "- Near stall/separation steady RANS itself is uncertain (`converged_coeffs` cases); expect larger errors there.",
           "- Only the GP provides a predictive uncertainty (±2σ bands); RF and MLP give point estimates.",
           ]
    if cfg.get("warnings"):
        L_ += [f"- {w}" for w in cfg["warnings"]]
    path.write_text("\n".join(L_) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="latest", help="'latest', a folder name under runs/surrogate, or a path")
    ap.add_argument("--data", type=Path, help="dataset CSV (default: the one recorded in config.yaml)")
    ap.add_argument("--aoa-col", default="aoa_deg")
    ap.add_argument("--cl-col", default="cl")
    ap.add_argument("--cd-col", default="cd")
    args = ap.parse_args(argv)

    import joblib
    import json
    import yaml
    try:
        run_dir = resolve_run(args.run)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    scalers = joblib.load(run_dir / "scalers.joblib")
    out = run_dir / "evaluation"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data_path = args.data or Path(cfg["data"]["path"])
    if not data_path.exists():
        print(f"ERROR: dataset not found: {data_path}", file=sys.stderr)
        return 1
    data_changed = sc.sha256(data_path) != cfg["data"]["sha256"]
    df, inputs, _ = sc.load_dataset(data_path, cfg["inputs"], cfg["targets"], cfg["group_col"], cfg["id_col"],
                                    cfg["status_filter"], drop_constant=False)
    print(f"Run: {run_dir}")
    print(f"Data: {data_path}" + ("  (CHANGED since training: only the recorded test rows are used)" if data_changed else ""))

    checks = leakage_checks(df, cfg, split, scalers)
    sc.dump_json(out / "leakage_check.json", checks)
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}")
    if not all(c["passed"] for c in checks):
        print("STOPPED: a data-leakage check failed; metrics would not reflect unseen geometries.", file=sys.stderr)
        for c in checks:
            if not c["passed"]:
                print(f"  - {c['check']}: {c['detail']}", file=sys.stderr)
        return 2

    models = sc.load_models(run_dir)
    if not models:
        print("ERROR: no models found in run", file=sys.stderr)
        return 1
    df_test = df[df[cfg["id_col"]].astype(str).isin(set(split["test_ids"]))].reset_index(drop=True)
    X = df_test[cfg["inputs"]].to_numpy(float)
    Y = df_test[cfg["targets"]].to_numpy(float)
    preds = {m: sc.predict(m, model, scalers, X) for m, model in models.items()}

    out_mask = sc.outside_range(X, cfg["inputs"], cfg["training_ranges"])
    met = all_metrics(Y, preds, cfg["targets"])
    met_in = all_metrics(Y, preds, cfg["targets"], ~out_mask) if out_mask.any() else {}
    met_out = all_metrics(Y, preds, cfg["targets"], out_mask) if out_mask.any() else {}
    # Best model: lowest mean RMSE normalized by the test std of each target
    best = min(met, key=lambda m: np.mean([met[m][t]["rmse"] / (Y[:, j].std() or 1)
                                           for j, t in enumerate(cfg["targets"])]))
    phys = physics_checks(df_test, models, scalers, cfg, args, cfg["seed"])
    figs = make_figures(out / "figures", Y, preds, X, cfg, df_test, models, scalers, args)

    sc.dump_json(out / "metrics.json", {
        "run": run_dir.name, "test_groups": split["test_groups"], "n_test_rows": int(len(df_test)),
        "data_changed_since_training": data_changed, "best_model": best, "metrics": met,
        "extrapolation": {"n_rows_outside_training_range": int(out_mask.sum()),
                          "ids": df_test.loc[out_mask, cfg["id_col"]].astype(str).tolist(),
                          "metrics_inside": met_in, "metrics_outside": met_out}})
    sc.dump_json(out / "physics_checks.json", phys)
    write_report(out / "report.md", run_dir, cfg, checks, met, met_in, met_out, int(out_mask.sum()), phys, figs,
                 X, Y, preds, df_test, best)

    print(f"\nTest set: {len(split['test_groups'])} unseen groups, {len(df_test)} rows")
    print(f"{'model':6} " + " ".join(f"{t + ' ' + k:>14}" for t in cfg["targets"] for k in ("MAE", "RMSE", "R2", "mean%")))
    for m, r in met.items():
        print(f"{m:6} " + " ".join(f"{fmt(r[t][k]):>14}" for t in cfg["targets"]
                                   for k in ("mae", "rmse", "r2", "mean_pct_error")))
    print(f"Best: {best}")
    if out_mask.any():
        print(f"WARNING: {int(out_mask.sum())} test rows are outside the training range (extrapolation)")
    for m, r in phys["cd_negative"].items():
        if r["test_points"] or r["random_in_range_points"]:
            print(f"WARNING: {m} predicts negative Cd ({r['test_points']} test, {r['random_in_range_points']} random points)")
    nflag = sum(len(v["flags"]) for e in phys["cl_vs_aoa"].values() for v in e["models"].values())
    if nflag:
        print(f"WARNING: {nflag} Cl(AoA) shape flag(s), see physics_checks.json")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
