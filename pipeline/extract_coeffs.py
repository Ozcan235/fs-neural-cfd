"""postProcessing/forceCoeffs1 içinden son N iterasyonun ortalama Cl/Cd değerini alıp CSV'ye satır ekler.

Kullanım:
    python -m pipeline.extract_coeffs --aoa 4 --h 0.2 --flap 20 [--last 300] [--csv data/dataset.csv]

Aynı case_id CSV'de zaten varsa satır güncellenir (yinelenen satır oluşmaz).
Not: liftDir = +y olduğundan Cl < 0 bastırma kuvveti (downforce) demektir.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from . import config

COLUMNS = ["case_id", "aoa_deg", "h_over_c", "flap_deg", "cl", "cd", "cl_std", "cd_std", "l_over_d",
           "n_avg", "iterations", "status", "cells", "u_inf", "re_c", "timestamp"]


def coeff_file(case_dir: Path) -> Path:
    base = case_dir / "postProcessing" / "forceCoeffs1"
    if not base.exists():
        raise FileNotFoundError(f"{base} yok (çözüm çalıştırılmamış veya forceCoeffs1 tanımlı değil)")
    files = sorted(base.glob("*/coefficient*.dat"), key=lambda f: float(f.parent.name)) or \
        sorted(base.glob("*/forceCoeffs*.dat"), key=lambda f: float(f.parent.name))
    if not files:
        raise FileNotFoundError(f"{base} içinde coefficient.dat bulunamadı")
    return files  # yeniden başlatmalarda birden fazla zaman klasörü olabilir


def read_coeffs(case_dir: Path) -> list[dict]:
    """Tüm coefficient.dat dosyalarını başlıktaki sütun adlarına göre okur, Time'a göre birleştirir."""
    rows = {}
    for f in coeff_file(case_dir):
        header = None
        for line in f.read_text(errors="replace").splitlines():
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                if cols and cols[0] == "Time":
                    header = cols
                continue
            if not header or not line.strip():
                continue
            vals = line.split()
            try:
                rec = {h: float(v) for h, v in zip(header, vals)}
            except ValueError:
                continue
            rows[rec["Time"]] = rec
    return [rows[t] for t in sorted(rows)]


def summarize(case_dir: Path, last: int) -> dict:
    data = read_coeffs(case_dir)
    if not data:
        raise ValueError("coefficient.dat boş")
    tail = data[-last:]

    def stats(key):
        v = [r[key] for r in tail]
        m = sum(v) / len(v)
        return m, (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

    cl, cl_std = stats("Cl")
    cd, cd_std = stats("Cd")
    return {"cl": cl, "cd": cd, "cl_std": cl_std, "cd_std": cd_std, "n_avg": len(tail), "iterations": int(data[-1]["Time"])}


def build_row(case_dir: Path, last: int) -> dict:
    params = json.loads((case_dir / "params.json").read_text())
    setup = json.loads((case_dir / "case_setup.json").read_text()) if (case_dir / "case_setup.json").exists() else {}
    status = json.loads((case_dir / "run_status.json").read_text())["status"] \
        if (case_dir / "run_status.json").exists() else "unchecked"
    mesh_q = json.loads((case_dir / "mesh_quality.json").read_text()) if (case_dir / "mesh_quality.json").exists() else {}
    s = summarize(case_dir, last)
    u = float(setup.get("U_INF", config.U_INF))
    return {
        "case_id": case_dir.name,
        "aoa_deg": params["aoa"], "h_over_c": params["h"], "flap_deg": params["flap"],
        "cl": f"{s['cl']:.6f}", "cd": f"{s['cd']:.6f}",
        "cl_std": f"{s['cl_std']:.2e}", "cd_std": f"{s['cd_std']:.2e}",
        "l_over_d": f"{s['cl'] / s['cd']:.4f}" if s["cd"] else "",
        "n_avg": s["n_avg"], "iterations": s["iterations"], "status": status,
        "cells": mesh_q.get("cells", ""), "u_inf": u, "re_c": f"{u * config.CHORD / float(setup.get('NU', config.NU)):.4g}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def upsert(csv_path: Path, row: dict) -> str:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    action = "eklendi"
    for i, r in enumerate(rows):
        if r.get("case_id") == row["case_id"]:
            rows[i] = row
            action = "güncellendi"
            break
    else:
        rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return action


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoa", type=float)
    ap.add_argument("--h", type=float)
    ap.add_argument("--flap", type=float)
    ap.add_argument("--case", type=Path)
    ap.add_argument("--last", type=int, default=config.AVG_LAST_N, help="Ortalama alınacak son iterasyon sayısı")
    ap.add_argument("--csv", type=Path, default=config.DATASET_CSV)
    ap.add_argument("--allow-unconverged", action="store_true",
                    help="run_status 'diverged'/'not_converged' olsa da yaz")
    args = ap.parse_args(argv)

    if args.case:
        case_dir = args.case.resolve()
    elif None not in (args.aoa, args.h, args.flap):
        case_dir = config.RUNS_DIR / config.case_id(args.aoa, args.h, args.flap)
    else:
        ap.error("--case veya --aoa/--h/--flap birlikte verilmeli")

    try:
        row = build_row(case_dir, args.last)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1
    if row["status"] in ("diverged", "not_converged") and not args.allow_unconverged:
        print(f"HATA: case durumu '{row['status']}'; veri setine yazılmadı (--allow-unconverged ile zorlayın).",
              file=sys.stderr)
        return 3
    action = upsert(args.csv, row)
    print(f"{row['case_id']}: Cl={row['cl']} (±{row['cl_std']})  Cd={row['cd']} (±{row['cd_std']})  "
          f"L/D={row['l_over_d']}  [son {row['n_avg']} iter, durum={row['status']}]")
    print(f"  {args.csv} -> satır {action}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
