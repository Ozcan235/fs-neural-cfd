"""Parametre ızgarasını mesh -> setup -> run -> extract adımlarından geçirir (veri seti üretimi).

Kullanım örnekleri:
    # Tam faktöriyel ızgara: başlangıç:bitiş:adet  veya  virgüllü liste
    python -m pipeline.sweep grid --aoa 0:12:4 --h 0.1,0.2,0.3 --flap 10:30:3
    # Quasi-random örnekleme (aralıklar config.PARAM_RANGES'tan, --aoa 0:12 ile değiştirilebilir)
    python -m pipeline.sweep sobol --n 64 --seed 0
    python -m pipeline.sweep lhs --n 100 --seed 1 --np 6
    # Surrogate eğitimi için önerilen: 32 (h, flap) konfigürasyonu x 7 AoA (grup bazlı bölme için)
    python -m pipeline.sweep sobol --n 32 --aoa-values 0:12:7
    # Sadece planı göster
    python -m pipeline.sweep sobol --n 16 --dry-run

Kaldığı yerden devam eder: dataset.csv'de başarılı satırı olan case'ler atlanır.
Iraksayan case bir kez --robust relaxation ile yeniden denenir.
Her adımın sonucu data/sweep_log.csv'ye yazılır.
"""
import argparse
import csv
import itertools
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config, extract_coeffs, mesh_wing, run_and_check, setup_case

LOG_COLUMNS = ["timestamp", "case_id", "aoa_deg", "h_over_c", "flap_deg", "stage", "result", "minutes"]
OK_STATUSES = {"converged", "converged_coeffs"}
ROUND = {"aoa": 2, "h": 3, "flap": 2}


def parse_axis(spec: str) -> list[float]:
    """'0:12:4' -> [0, 4, 8, 12];  '0.1,0.2' -> [0.1, 0.2]"""
    if ":" in spec:
        a, b, n = spec.split(":")
        a, b, n = float(a), float(b), int(n)
        return [a + (b - a) * i / (n - 1) for i in range(n)] if n > 1 else [a]
    return [float(x) for x in spec.split(",")]


def parse_range(spec: str | None, name: str) -> tuple[float, float]:
    if not spec:
        return config.PARAM_RANGES[name]
    a, b = spec.split(":")[:2]
    return float(a), float(b)


def make_plan(args) -> list[dict]:
    names = ("aoa", "h", "flap")
    if args.method == "grid":
        axes = [parse_axis(getattr(args, n) or "{}:{}:3".format(*config.PARAM_RANGES[n])) for n in names]
        pts = list(itertools.product(*axes))
    else:
        from scipy.stats import qmc
        # --aoa-values verilirse yalnızca (h, flap) konfigürasyonları örneklenir, her biri AoA listesiyle taranır
        sampled = ("h", "flap") if args.aoa_values else names
        d = len(sampled)
        sampler = (qmc.Sobol(d=d, scramble=True, seed=args.seed) if args.method == "sobol"
                   else qmc.LatinHypercube(d=d, seed=args.seed))
        unit = sampler.random(args.n)
        lo, hi = zip(*(parse_range(getattr(args, n), n) for n in sampled))
        pts = qmc.scale(unit, lo, hi).tolist()
        if args.aoa_values:
            pts = [(a, h, f) for h, f in pts for a in parse_axis(args.aoa_values)]
    plan, seen = [], set()
    for p in pts:
        d = {n: round(v, ROUND[n]) for n, v in zip(names, p)}
        cid = config.case_id(**d)
        if cid not in seen:
            seen.add(cid)
            plan.append({"case_id": cid, **d})
    return plan


def done_cases(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {r["case_id"] for r in csv.DictReader(f) if r.get("status") in OK_STATUSES}


def log(path: Path, p: dict, stage: str, result: str, t0: float) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), "case_id": p["case_id"],
                    "aoa_deg": p["aoa"], "h_over_c": p["h"], "flap_deg": p["flap"], "stage": stage,
                    "result": result, "minutes": f"{(time.time() - t0) / 60:.1f}"})


def run_one(p: dict, args, log_path: Path) -> str:
    a, h, f = (str(p[k]) for k in ("aoa", "h", "flap"))
    ids = ["--aoa", a, "--h", h, "--flap", f]
    t0 = time.time()

    code = mesh_wing.main(ids + ["--force"] + (["--set", *args.set] if args.set else []))
    log(log_path, p, "mesh", {0: "ok", 2: "quality_failed"}.get(code, "error"), t0)
    if code == 1 or (code == 2 and not args.allow_bad_mesh):
        return "mesh_failed"

    for attempt, robust in enumerate((False, True)):
        t1 = time.time()
        setup_args = ids + ["--np", str(args.np), "--end-time", str(args.end_time)] + (["--robust"] if robust else [])
        if setup_case.main(setup_args) != 0:
            log(log_path, p, "setup", "error", t1)
            return "setup_failed"
        code = run_and_check.main(ids)
        status = {0: "converged", 3: "not_converged", 4: "diverged"}.get(code, "error")
        log(log_path, p, "run" + ("_robust" if robust else ""), status, t1)
        if status != "diverged" or attempt == 1:
            break
        print("  -> ıraksadı, --robust ile yeniden deneniyor")

    t2 = time.time()
    ex = extract_coeffs.main(ids + ["--csv", str(args.csv), "--last", str(args.last)]
                             + (["--allow-unconverged"] if args.keep_unconverged else []))
    log(log_path, p, "extract", "ok" if ex == 0 else "skipped", t2)

    if args.cleanup:
        case_dir = config.RUNS_DIR / p["case_id"]
        for d in case_dir.glob("processor*"):
            shutil.rmtree(d, ignore_errors=True)
    return status if ex == 0 else f"{status}_not_saved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=["grid", "sobol", "lhs"])
    ap.add_argument("--aoa", help="grid: 'a:b:n' veya liste; sobol/lhs: 'a:b' aralığı")
    ap.add_argument("--h")
    ap.add_argument("--flap")
    ap.add_argument("--n", type=int, default=32, help="sobol/lhs örnek sayısı (sobol için 2'nin kuvveti önerilir)")
    ap.add_argument("--aoa-values", help="sobol/lhs: (h, flap) konfigürasyonlarını örnekle, her birini bu AoA "
                                         "listesiyle tara ('0:12:7' veya '0,4,8'). --n konfigürasyon sayısı olur")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--np", type=int, default=config.N_PROCS)
    ap.add_argument("--end-time", type=int, default=config.END_TIME)
    ap.add_argument("--last", type=int, default=config.AVG_LAST_N)
    ap.add_argument("--csv", type=Path, default=config.DATASET_CSV)
    ap.add_argument("--set", nargs="*", default=[], metavar="AD=DEGER", help="wing.geo sabitleri (tüm case'ler)")
    ap.add_argument("--max-cases", type=int, help="Bu çalıştırmada en fazla kaç case çözülsün")
    ap.add_argument("--allow-bad-mesh", action="store_true")
    ap.add_argument("--keep-unconverged", action="store_true", help="Yakınsamayanları da CSV'ye yaz")
    ap.add_argument("--cleanup", action="store_true", help="Bitince processor* klasörlerini sil")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    plan = make_plan(args)
    plan_path = config.DATA_DIR / f"sweep_plan_{args.method}.csv"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with plan_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "aoa", "h", "flap"])
        w.writeheader()
        w.writerows(plan)

    done = done_cases(args.csv)
    todo = [p for p in plan if p["case_id"] not in done]
    if args.max_cases:
        todo = todo[:args.max_cases]
    print(f"Plan: {len(plan)} case ({plan_path}), tamamlanmış: {len(plan) - len([p for p in plan if p['case_id'] not in done])}, "
          f"bu çalıştırmada: {len(todo)}")
    if args.dry_run:
        for p in todo:
            print(f"  {p['case_id']}  aoa={p['aoa']}  h={p['h']}  flap={p['flap']}")
        return 0

    log_path = config.DATA_DIR / "sweep_log.csv"
    summary, t_start = {}, time.time()
    for i, p in enumerate(todo, 1):
        print(f"\n=== [{i}/{len(todo)}] {p['case_id']} ===", flush=True)
        try:
            result = run_one(p, args, log_path)
        except Exception as e:  # tek bir case tüm sweep'i durdurmasın
            print(f"  BEKLENMEYEN HATA: {e}", file=sys.stderr)
            log(log_path, p, "exception", str(e)[:200], time.time())
            result = "exception"
        summary[result] = summary.get(result, 0) + 1
        elapsed = (time.time() - t_start) / 60
        print(f"  sonuç: {result}  | geçen {elapsed:.0f} dk, tahmini kalan {elapsed / i * (len(todo) - i):.0f} dk",
              flush=True)

    print("\nÖzet: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))
    print(f"Veri seti: {args.csv}   Log: {log_path}")
    return 0 if all(k in OK_STATUSES for k in summary) else 3


if __name__ == "__main__":
    sys.exit(main())
