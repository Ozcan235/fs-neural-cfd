"""Parametrik wing.geo'dan mesh üretir, OpenFOAM'a çevirir ve kalitesini denetler.

Kullanım:
    python -m pipeline.mesh_wing --aoa 4 --h 0.2 --flap 20 [--set gap=0.025 ...] [--force]

Adımlar: gmsh (Windows, .venv) -> gmshToFoam -> sınır tiplerini düzelt -> checkMesh -> eşik kontrolü.
Çıktı: runs/<case_id>/ (wing.msh, constant/polyMesh, log.*, mesh_quality.json)
Çıkış kodu: 0 = mesh geçerli, 2 = kalite eşikleri aşıldı, 1 = hata.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config, foam
from .setup_case import write_system

PATCH_TYPES = {"frontAndBack": "empty", "ground": "wall", "wing_main": "wall", "wing_flap": "wall"}
EXPECTED_PATCHES = {"inlet", "outlet", "top", "ground", "wing_main", "wing_flap", "frontAndBack"}

# -allGeometry kontrollerinden yalnızca bilgi amaçlı olanlar (eşik kontrolü ayrıca yapılır)
ADVISORY = ("small interpolation weight", "small volume ratio", "face flatness", "concave", "determinant")

GMSH_RUNNER = "import sys, gmsh; gmsh.initialize(sys.argv, run=True); gmsh.finalize()"


def check_ranges(aoa: float, h: float, flap: float) -> list[str]:
    msgs = []
    for name, val in (("aoa", aoa), ("h", h), ("flap", flap)):
        lo, hi = config.PARAM_RANGES[name]
        if not lo <= val <= hi:
            msgs.append(f"{name}={val} tanımlı aralığın dışında [{lo}, {hi}]")
    return msgs


def run_gmsh(case_dir: Path, aoa: float, h: float, flap: float, extra: dict[str, float]) -> None:
    msh = case_dir / "wing.msh"
    numbers = {"AoA": aoa, "hc": h, "Flap": flap, "c": config.CHORD, "span": config.SPAN, **extra}
    cmd = [sys.executable, "-c", GMSH_RUNNER, str(config.GEO_FILE), "-3", "-format", "msh22", "-o", str(msh)]
    for k, v in numbers.items():
        cmd += ["-setnumber", k, str(v)]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    (case_dir / "log.gmsh").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    errors = [l for l in (proc.stdout + proc.stderr).splitlines() if l.startswith("Error")]
    if proc.returncode != 0 or errors or not msh.exists():
        raise RuntimeError("gmsh başarısız:\n" + "\n".join(errors[-10:] or (proc.stdout + proc.stderr).splitlines()[-15:]))


def fix_boundary(case_dir: Path) -> dict[str, int]:
    """gmshToFoam tüm yamaları 'patch' yapar; doğru tipleri ata, yüz sayılarını döndür."""
    foam.run(case_dir, " && ".join(
        f"foamDictionary constant/polyMesh/boundary -entry entry0/{patch}/type -set {ptype} >/dev/null"
        for patch, ptype in PATCH_TYPES.items()))
    text = (case_dir / "constant" / "polyMesh" / "boundary").read_text(errors="replace")
    faces = {m.group(1): int(m.group(2)) for m in
             re.finditer(r"^\s*(\w+)\s*\{[^}]*?nFaces\s+(\d+);", text, re.M | re.S)}
    missing = EXPECTED_PATCHES - faces.keys()
    empty = [p for p in EXPECTED_PATCHES & faces.keys() if faces[p] == 0]
    if missing or empty:
        raise RuntimeError(f"Beklenen yamalar eksik/boş: eksik={sorted(missing)} boş={empty}. Mevcut: {faces}")
    return faces


def parse_checkmesh(log: str) -> dict:
    def grab(pattern, cast=float):
        m = re.search(pattern, log)
        return cast(m.group(1)) if m else None

    stars = [l.strip() for l in log.splitlines() if l.strip().startswith("***")]
    return {
        "cells": grab(r"(?m)^\s*cells:\s+(\d+)", int),
        "max_non_orthogonality": grab(r"non-orthogonality Max:\s*([\d.eE+-]+)"),
        "avg_non_orthogonality": grab(r"non-orthogonality Max:\s*[\d.eE+-]+\s+average:\s*([\d.eE+-]+)"),
        "max_skewness": grab(r"Max skewness = ([\d.eE+-]+)"),
        "max_aspect_ratio": grab(r"Max aspect ratio = ([\d.eE+-]+)"),
        "min_determinant": grab(r"Cell determinant \(wellposedness\) : minimum: ([\d.eE+-]+)"),
        "failed_checks": grab(r"Failed (\d+) mesh checks", int) or 0,
        "errors": [l.strip("* ") for l in stars if not any(a in l for a in ADVISORY)],
        "warnings": [l.strip("* ") for l in stars if any(a in l for a in ADVISORY)],
        "mesh_ok": "Mesh OK." in log,
    }


def evaluate(q: dict) -> list[str]:
    lim = config.MESH_LIMITS
    problems = []
    if q["max_non_orthogonality"] is not None and q["max_non_orthogonality"] > lim["max_non_orthogonality"]:
        problems.append(f"non-orthogonality {q['max_non_orthogonality']:.1f} > {lim['max_non_orthogonality']}")
    if q["max_skewness"] is not None and q["max_skewness"] > lim["max_skewness"]:
        problems.append(f"skewness {q['max_skewness']:.2f} > {lim['max_skewness']}")
    if q["max_aspect_ratio"] is not None and q["max_aspect_ratio"] > lim["max_aspect_ratio"]:
        problems.append(f"aspect ratio {q['max_aspect_ratio']:.0f} > {lim['max_aspect_ratio']}")
    if q["min_determinant"] is not None and q["min_determinant"] < lim["min_determinant"]:
        problems.append(f"cell determinant {q['min_determinant']:.2e} < {lim['min_determinant']}")
    problems += [f"checkMesh: {e}" for e in q["errors"]]
    return problems


def mesh(aoa: float, h: float, flap: float, extra: dict[str, float] | None = None,
         case_dir: Path | None = None) -> tuple[Path, dict, list[str]]:
    case_dir = case_dir or config.RUNS_DIR / config.case_id(aoa, h, flap)
    if case_dir.exists():
        for name in ("constant/polyMesh", "wing.msh"):
            p = case_dir / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "constant").mkdir(exist_ok=True)
    write_system(case_dir)
    params = {"aoa": aoa, "h": h, "flap": flap, "chord": config.CHORD, "span": config.SPAN, **(extra or {})}
    (case_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    run_gmsh(case_dir, aoa, h, flap, extra or {})
    foam.run(case_dir, "gmshToFoam wing.msh", log_name="gmshToFoam")
    faces = fix_boundary(case_dir)
    foam.run(case_dir, "checkMesh -allGeometry -allTopology", log_name="checkMesh", check=False)
    q = parse_checkmesh((case_dir / "log.checkMesh").read_text(errors="replace"))
    q["patch_faces"] = faces
    problems = evaluate(q)
    q["problems"] = problems
    q["passed"] = not problems
    (case_dir / "mesh_quality.json").write_text(json.dumps(q, indent=2), encoding="utf-8")
    return case_dir, q, problems


def parse_set(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, _, v = it.partition("=")
        out[k.strip()] = float(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoa", type=float, required=True, help="Ana eleman hücum açısı [deg]")
    ap.add_argument("--h", type=float, required=True, help="Yerden yükseklik h/c (en alt nokta)")
    ap.add_argument("--flap", type=float, required=True, help="Flap açısı [deg], ana elemana göre")
    ap.add_argument("--set", nargs="*", default=[], metavar="AD=DEGER",
                    help="wing.geo içindeki diğer sabitler, örn. gap=0.025 sWing=0.003")
    ap.add_argument("--force", action="store_true", help="Parametre aralığı dışında da çalıştır")
    args = ap.parse_args(argv)

    warn = check_ranges(args.aoa, args.h, args.flap)
    if warn and not args.force:
        print("HATA: " + "; ".join(warn) + " (--force ile zorlayabilirsiniz)", file=sys.stderr)
        return 1
    try:
        case_dir, q, problems = mesh(args.aoa, args.h, args.flap, parse_set(args.set))
    except RuntimeError as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    print(f"Mesh: {case_dir}")
    print(f"  hücre={q['cells']}  non-ortho max={q['max_non_orthogonality']} (ort {q['avg_non_orthogonality']})"
          f"  skewness max={q['max_skewness']}  aspect max={q['max_aspect_ratio']}  det min={q['min_determinant']}")
    print(f"  yamalar: {q['patch_faces']}")
    for w in q["warnings"]:
        print(f"  uyarı: {w}")
    if problems:
        print("KALİTE EŞİĞİ AŞILDI: " + "; ".join(problems))
        return 2
    print("Mesh kalite eşiklerini geçti.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
