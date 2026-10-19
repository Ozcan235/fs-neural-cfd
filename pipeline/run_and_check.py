"""simpleFoam'u çalıştırır, log'dan residual'ları ve Cl/Cd kararlılığını okuyup yakınsamayı değerlendirir.

Kullanım:
    python -m pipeline.run_and_check --aoa 4 --h 0.2 --flap 20 [--np 4] [--no-run]

--no-run: çözücüyü çalıştırmadan mevcut log.simpleFoam'u analiz eder.
Çıktı: runs/<case_id>/run_status.json
Çıkış kodu: 0 = yakınsadı, 3 = yakınsamadı (bitti ama kriter sağlanmadı), 4 = ıraksadı/çöktü, 1 = hata.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

from . import config, foam
from .extract_coeffs import read_coeffs

FIELDS = ("Ux", "Uy", "p", "k", "omega")
RE_TIME = re.compile(r"^Time = (\S+)", re.M)
RE_SOLVE = re.compile(r"Solving for (\w+), Initial residual = ([\deE.+-]+|nan|-?inf)", re.I)


def parse_log(text: str) -> dict:
    """Her iterasyon için ilk initial residual'ı toplar."""
    iters, res, bounding = [], {f: [] for f in FIELDS}, []
    blocks = RE_TIME.split(text)
    # blocks: [önsöz, t1, blok1, t2, blok2, ...]
    for t, block in zip(blocks[1::2], blocks[2::2]):
        seen = {}
        for f, v in RE_SOLVE.findall(block):
            if f in res and f not in seen:
                seen[f] = float(v)
        bounding.append(re.findall(r"bounding (\w+), min:", block))
        if seen:
            iters.append(float(t))
            for f in FIELDS:
                res[f].append(seen.get(f))
    # İlk iterasyonlardaki bounding normaldir; yalnızca çözümün ikinci yarısındakiler sayılır
    late = [f for b in bounding[len(bounding) // 2:] for f in b]
    cont = re.findall(r"time step continuity errors : sum local = ([\deE.+-]+)", text)
    return {
        "iters": iters,
        "res": res,
        "bounding_count": len(late),
        "bounding_fields": sorted(set(late)),
        "last_continuity": float(cont[-1]) if cont else None,
        # Açılıştaki "Floating point exception trapping enabled" satırı çökme değildir
        "fpe": bool(re.search(r"Floating point exception(?! trapping)|printStack|FOAM FATAL", text)),
        "fatal_msg": (re.search(r"(--> FOAM FATAL.*?)(?:\n\n|\Z)", text, re.S) or [None, None])[1],
        "converged_msg": "reached convergence criteria" in text or "SIMPLE solution converged" in text,
        "ended": bool(re.search(r"^End\s*$", text, re.M)),
    }


def coeff_stability(case_dir: Path, window: int) -> dict:
    try:
        data = read_coeffs(case_dir)
    except FileNotFoundError:
        return {"available": False}
    tail = data[-window:]
    out = {"available": True, "n": len(data)}
    for name in ("Cl", "Cd"):
        vals = [r[name] for r in tail if math.isfinite(r[name])]
        if not vals:
            out[name] = None
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        # Pencerenin ilk ve son yarısı arasındaki kayma (drift)
        half = len(vals) // 2 or 1
        drift = sum(vals[half:]) / max(len(vals) - half, 1) - sum(vals[:half]) / half
        out[name] = {"mean": mean, "std": std, "rel_std": std / abs(mean) if mean else math.inf,
                     "drift": drift, "rel_drift": abs(drift) / abs(mean) if mean else math.inf}
    return out


def diagnose(log: dict, stab: dict, mesh_q: dict | None, end_time: int, robust: bool) -> tuple[str, list[str]]:
    """Durum ve okunabilir sebep/öneri listesi döndürür."""
    reasons = []
    recent = [x for v in log["res"].values() for x in v[-5:] if x is not None]
    has_nan = any(not math.isfinite(x) for x in recent)

    if log["fpe"] or has_nan or (log["fatal_msg"] and not log["ended"]):
        status = "diverged"
        reasons.append("Çözüm çöktü (floating point exception / NaN / FOAM FATAL).")
        if log["fatal_msg"]:
            reasons.append("Hata mesajı: " + log["fatal_msg"].strip().splitlines()[0][:300])
        it = len(log["iters"])
        if it < 50:
            reasons.append(f"İlk {it} iterasyonda çöktü: büyük olasılıkla sınır koşulu, başlangıç alanı veya mesh hatası.")
        else:
            reasons.append(f"{it}. iterasyonda çöktü: relaxation çok agresif olabilir (setup-case --robust deneyin).")
    elif log["converged_msg"]:
        status = "converged"
    else:
        status = "not_converged"

    if log["bounding_count"]:
        reasons.append(f"{log['bounding_count']} kez 'bounding' uyarısı ({', '.join(log['bounding_fields'])}): "
                       "çözümün ikinci yarısında türbülans alanları negatif oluyor; düşük kaliteli hücre veya kararsızlık belirtisi.")
    if log["last_continuity"] and log["last_continuity"] > 1e-3:
        reasons.append(f"Süreklilik hatası yüksek ({log['last_continuity']:.2e}).")

    # Residual eğilimi: son 200 iterasyonda artış var mı?
    for f, v in log["res"].items():
        vv = [x for x in v if x is not None and math.isfinite(x)]
        if len(vv) > 400:
            a, b = sum(vv[-400:-200]) / 200, sum(vv[-200:]) / 200
            if b > 3 * a:
                reasons.append(f"{f} residual'ı son iterasyonlarda artıyor ({a:.2e} -> {b:.2e}).")

    coeff_ok = False
    if stab.get("available") and stab.get("Cl") and stab.get("Cd"):
        cl, cd = stab["Cl"], stab["Cd"]
        coeff_ok = cl["rel_std"] < config.CONV_COEFF_REL_STD and cd["rel_std"] < config.CONV_COEFF_REL_STD
        if status != "diverged" and not coeff_ok:
            reasons.append(f"Son {config.CONV_WINDOW} iterasyonda Cl/Cd oynuyor "
                           f"(Cl rel_std={cl['rel_std']:.2%}, Cd rel_std={cd['rel_std']:.2%}); "
                           "akış ayrılması / periyodik girdap kopması olabilir (yüksek flap açısı veya çok düşük h).")
    if status == "not_converged" and coeff_ok:
        # Residual eşiğine inmedi ama integral büyüklükler kararlı: veri için kabul edilebilir
        status = "converged_coeffs"
        reasons.append(f"Residual'lar {config.CONV_RESIDUAL:g} eşiğine inmedi ama Cl/Cd kararlı; katsayılar kullanılabilir.")
    elif status == "not_converged":
        n = len(log["iters"])
        reasons.append(f"{n}/{end_time} iterasyon, residual kriteri sağlanmadı. "
                       + ("endTime artırılabilir." if n >= end_time else "Çözücü erken durdu."))

    if mesh_q and status not in ("converged", "converged_coeffs"):
        if mesh_q.get("max_non_orthogonality", 0) > 60:
            reasons.append(f"Mesh non-orthogonality yüksek ({mesh_q['max_non_orthogonality']:.0f}).")
        if mesh_q.get("warnings"):
            reasons.append("Mesh uyarıları: " + "; ".join(mesh_q["warnings"]))
    if status == "diverged" and not robust:
        reasons.append("Öneri: /setup-case ... --robust ile düşük relaxation kullanıp tekrar çalıştırın.")
    return status, reasons


def run_solver(case_dir: Path, n_procs: int) -> None:
    if n_procs > 1:
        foam.run(case_dir, "decomposePar -force", log_name="decomposePar")
        foam.run(case_dir, f"mpirun -np {n_procs} simpleFoam -parallel", log_name="simpleFoam", check=False)
        foam.run(case_dir, "reconstructPar -latestTime", log_name="reconstructPar", check=False)
    else:
        foam.run(case_dir, "simpleFoam", log_name="simpleFoam", check=False)


def analyze(case_dir: Path) -> dict:
    setup = json.loads((case_dir / "case_setup.json").read_text()) if (case_dir / "case_setup.json").exists() else {}
    mesh_q = json.loads((case_dir / "mesh_quality.json").read_text()) if (case_dir / "mesh_quality.json").exists() else None
    log_path = case_dir / "log.simpleFoam"
    if not log_path.exists():
        raise FileNotFoundError(f"{log_path} yok")
    log = parse_log(log_path.read_text(errors="replace"))
    stab = coeff_stability(case_dir, config.CONV_WINDOW)
    end_time = int(setup.get("END_TIME", config.END_TIME))
    status, reasons = diagnose(log, stab, mesh_q, end_time, robust=setup.get("RELAX_P", 1) < 1)
    final = {f: (v[-1] if v else None) for f, v in log["res"].items()}
    result = {
        "status": status,
        "iterations": len(log["iters"]),
        "final_residuals": final,
        "coeffs_window": stab,
        "bounding_count": log["bounding_count"],
        "reasons": reasons,
    }
    (case_dir / "run_status.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


EXIT = {"converged": 0, "converged_coeffs": 0, "not_converged": 3, "diverged": 4}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoa", type=float)
    ap.add_argument("--h", type=float)
    ap.add_argument("--flap", type=float)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--np", type=int, help="İşlemci sayısı (varsayılan: case_setup.json'daki N_PROCS)")
    ap.add_argument("--no-run", action="store_true", help="Sadece mevcut log'u analiz et")
    args = ap.parse_args(argv)

    if args.case:
        case_dir = args.case.resolve()
    elif None not in (args.aoa, args.h, args.flap):
        case_dir = config.RUNS_DIR / config.case_id(args.aoa, args.h, args.flap)
    else:
        ap.error("--case veya --aoa/--h/--flap birlikte verilmeli")
    if not (case_dir / "0").exists():
        print(f"HATA: {case_dir} kurulmamış. Önce /setup-case çalıştırın.", file=sys.stderr)
        return 1

    if not args.no_run:
        setup = json.loads((case_dir / "case_setup.json").read_text())
        n = args.np or int(setup.get("N_PROCS", 1))
        t0 = time.time()
        print(f"simpleFoam çalışıyor: {case_dir} (np={n}) ...", flush=True)
        try:
            run_solver(case_dir, n)
        except RuntimeError as e:
            print(f"HATA: {e}", file=sys.stderr)
            return 1
        print(f"  süre: {(time.time() - t0) / 60:.1f} dk")

    try:
        r = analyze(case_dir)
    except FileNotFoundError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    print(f"Durum: {r['status']}  iterasyon: {r['iterations']}  bounding: {r['bounding_count']}")
    print("  son residual'lar: " + "  ".join(
        f"{k}={v:.2e}" for k, v in r["final_residuals"].items() if v is not None and math.isfinite(v)))
    s = r["coeffs_window"]
    if s.get("available") and s.get("Cl"):
        print(f"  son {config.CONV_WINDOW} iter: Cl={s['Cl']['mean']:.4f} (±{s['Cl']['std']:.1e})  "
              f"Cd={s['Cd']['mean']:.5f} (±{s['Cd']['std']:.1e})")
    for reason in r["reasons"]:
        print(f"  - {reason}")
    return EXIT[r["status"]]


if __name__ == "__main__":
    sys.exit(main())
