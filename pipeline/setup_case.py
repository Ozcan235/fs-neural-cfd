"""Şablon case'i kopyalar ve @@...@@ yer tutucularını doldurur.

Kullanım:
    python -m pipeline.setup_case --aoa 4 --h 0.2 --flap 20 [--u 20] [--end-time 3000] [--np 1] [--robust]

Mesh (constant/polyMesh) önceden /mesh-wing ile aynı klasörde üretilmiş olmalıdır.
"""
import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

from . import config

PLACEHOLDER = re.compile(r"@@([A-Z_]+)@@")

RELAX_NORMAL = {"RELAX_P": 1.0, "RELAX_U": 0.9, "RELAX_TURB": 0.7}
RELAX_ROBUST = {"RELAX_P": 0.3, "RELAX_U": 0.5, "RELAX_TURB": 0.4}


def build_values(u_inf=config.U_INF, end_time=config.END_TIME, n_procs=config.N_PROCS,
                 robust=False, intensity=config.TURB_INTENSITY, visc_ratio=config.TURB_VISC_RATIO,
                 nu=config.NU, res_tol=config.CONV_RESIDUAL) -> dict:
    k = 1.5 * (u_inf * intensity) ** 2
    omega = k / (nu * visc_ratio)
    values = {
        "U_INF": u_inf,
        "NU": nu,
        "K_INF": f"{k:.6g}",
        "OMEGA_INF": f"{omega:.6g}",
        "RHO_INF": config.RHO_INF,
        "L_REF": config.CHORD,
        "A_REF": f"{config.CHORD * config.SPAN:.6g}",
        "COFR_X": 0.25 * config.CHORD,
        "PATCHES": "wing_main wing_flap",
        "END_TIME": int(end_time),
        "WRITE_INTERVAL": min(config.WRITE_INTERVAL, int(end_time)),
        "N_PROCS": int(n_procs),
        "RES_TOL": res_tol,
    }
    values.update(RELAX_ROBUST if robust else RELAX_NORMAL)
    return values


def render_tree(src: Path, dst: Path, values: dict) -> None:
    """src altındaki dosyaları dst'ye kopyalar, yer tutucuları doldurur."""
    for f in src.rglob("*"):
        if f.is_dir():
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = f.read_text(encoding="utf-8")

        def sub(m):
            key = m.group(1)
            if key not in values:
                raise KeyError(f"{f}: '{key}' yer tutucusu için değer yok")
            return str(values[key])

        out.write_text(PLACEHOLDER.sub(sub, text), encoding="utf-8", newline="\n")


def write_system(case_dir: Path, values: dict | None = None) -> None:
    """Sadece system/ klasörünü yazar (gmshToFoam/checkMesh için gerekli)."""
    render_tree(config.TEMPLATE_CASE / "system", case_dir / "system", values or build_values())


def setup(case_dir: Path, values: dict, clean: bool = True) -> None:
    if not (case_dir / "constant" / "polyMesh" / "boundary").exists():
        raise FileNotFoundError(f"{case_dir} içinde constant/polyMesh yok. Önce /mesh-wing çalıştırın.")
    if clean:
        # Eski çözüm ve sonuçları temizle (mesh korunur)
        for d in case_dir.iterdir():
            if d.is_dir() and (re.fullmatch(r"\d+(\.\d+)?", d.name)
                               or d.name.startswith("processor") or d.name == "postProcessing"):
                shutil.rmtree(d)
        for f in ("log.simpleFoam", "log.decomposePar", "log.reconstructPar", "run_status.json"):
            (case_dir / f).unlink(missing_ok=True)
    for sub in ("0", "constant", "system"):
        render_tree(config.TEMPLATE_CASE / sub, case_dir / sub, values)
    (case_dir / "case_setup.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoa", type=float)
    ap.add_argument("--h", type=float)
    ap.add_argument("--flap", type=float)
    ap.add_argument("--case", type=Path, help="Doğrudan case klasörü (aoa/h/flap yerine)")
    ap.add_argument("--u", type=float, default=config.U_INF, help="Serbest akış hızı [m/s]")
    ap.add_argument("--end-time", type=int, default=config.END_TIME)
    ap.add_argument("--np", type=int, default=config.N_PROCS, help="İşlemci sayısı")
    ap.add_argument("--robust", action="store_true", help="Düşük relaxation (ıraksayan case'ler için)")
    ap.add_argument("--intensity", type=float, default=config.TURB_INTENSITY)
    args = ap.parse_args(argv)

    if args.case:
        case_dir = args.case.resolve()
    elif None not in (args.aoa, args.h, args.flap):
        case_dir = config.RUNS_DIR / config.case_id(args.aoa, args.h, args.flap)
    else:
        ap.error("--case veya --aoa/--h/--flap birlikte verilmeli")

    values = build_values(u_inf=args.u, end_time=args.end_time, n_procs=args.np,
                          robust=args.robust, intensity=args.intensity)
    try:
        setup(case_dir, values)
    except (FileNotFoundError, KeyError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    re_c = args.u * config.CHORD / config.NU
    print(f"Case hazır: {case_dir}")
    print(f"  U={args.u} m/s  Re_c={re_c:.3g}  k={values['K_INF']}  omega={values['OMEGA_INF']}")
    print(f"  forceCoeffs: lRef={values['L_REF']} Aref={values['A_REF']} patches=({values['PATCHES']})")
    print(f"  endTime={values['END_TIME']} np={values['N_PROCS']} relaxation={'robust' if args.robust else 'normal'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
