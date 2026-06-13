#!/usr/bin/env python3
"""
Springboks Rugby Analytics — Predictive Pipeline
University of Bern · CAS Applied Data Science
Author: Tamara Brand

Executes all 6 notebooks in sequence (00 → 05).

Usage:
    python3 run_pipeline.py                    # full pipeline
    python3 run_pipeline.py --from 02          # restart from NB02
    python3 run_pipeline.py --only 05          # run only NB05
    python3 run_pipeline.py --from 02 --to 04  # run NB02, 03, 04

Outputs:
    data/bronze/  → 4 parquet files (raw sources)
    data/silver/  → silver_results.parquet
    data/gold/    → gold_train.parquet, gold_test.parquet
    models/       → best_model.pkl
    figures/      → PDF visualisations
    reports/      → eda_summary.csv, model_comparison.csv,
                    walk_forward_cv.csv, rwc2027_results.csv
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
PYTHON = ROOT / ".venv" / "bin" / "python3"
KERNEL = "springboks-venv"

NOTEBOOKS = [
    ("00", "notebooks/00_data_ingestion.ipynb"),
    ("01", "notebooks/01_data_cleaning.ipynb"),
    ("02", "notebooks/02_feature_engineering.ipynb"),
    ("03", "notebooks/03_exploratory_analysis.ipynb"),
    ("04", "notebooks/04_model_training.ipynb"),
    ("05", "notebooks/05_model_application.ipynb"),
]


# ── helpers ──────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    if seconds < 60:   return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def run_notebook(nb_id: str, path: str, idx: int, total: int) -> bool:
    nb_path = ROOT / path
    if not nb_path.exists():
        print(f"  ❌  NOT FOUND: {nb_path}")
        return False

    print(f"\n[{idx}/{total}] {nb_path.name}")
    print("─" * 60)

    start  = datetime.now()
    result = subprocess.run([
        str(PYTHON), "-m", "jupyter", "nbconvert",
        "--to",      "notebook",
        "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL}",
        "--inplace",
        str(nb_path),
    ])
    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        print(f"  ❌  FAILED after {fmt_time(elapsed)}")
        return False

    print(f"  ✅  Done in {fmt_time(elapsed)}")
    return True


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Springboks Rugby Analytics Pipeline"
    )
    parser.add_argument("--from", dest="start", default="00",
                        help="Start from notebook ID (default: 00)")
    parser.add_argument("--to",   dest="end",   default="05",
                        help="Run up to notebook ID (default: 05)")
    parser.add_argument("--only", dest="only",  default=None,
                        help="Run only one notebook, e.g. --only 05")
    args = parser.parse_args()

    # select notebooks
    if args.only:
        to_run = [(i, n) for i, n in NOTEBOOKS if i == args.only]
    else:
        to_run = [(i, n) for i, n in NOTEBOOKS
                  if args.start <= i <= args.end]

    if not to_run:
        print(f"No notebooks matched. Available: {[i for i,_ in NOTEBOOKS]}")
        sys.exit(1)

    # validate files exist
    missing = [n for _, n in to_run if not (ROOT / n).exists()]
    if missing:
        print("ERROR — missing notebooks:")
        for m in missing: print(f"  {ROOT / m}")
        sys.exit(1)

    # validate kernel
    check = subprocess.run(
        [str(PYTHON), "-m", "jupyter", "kernelspec", "list"],
        capture_output=True, text=True
    )
    if KERNEL not in check.stdout:
        print(f"ERROR: kernel '{KERNEL}' not found.")
        print(f"Register it with:")
        print(f"  python3 -m ipykernel install --user "
              f"--name={KERNEL} --display-name='Python ({KERNEL})'")
        sys.exit(1)

    # header
    print("\n" + "=" * 60)
    print("  SPRINGBOKS RUGBY ANALYTICS PIPELINE")
    print("=" * 60)
    print(f"  Start:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Notebooks: {len(to_run)}  ({to_run[0][0]} → {to_run[-1][0]})")
    print(f"  Kernel:    {KERNEL}")
    print(f"  Root:      {ROOT}")
    print("=" * 60)

    # run
    t0 = datetime.now()
    for idx, (nb_id, nb_path) in enumerate(to_run, 1):
        if not run_notebook(nb_id, nb_path, idx, len(to_run)):
            print(f"\n{'=' * 60}")
            print(f"  PIPELINE STOPPED at NB{nb_id}")
            print(f"  Fix the error, then restart with:")
            print(f"  python3 run_pipeline.py --from {nb_id}")
            print("=" * 60)
            sys.exit(1)

    elapsed = (datetime.now() - t0).total_seconds()

    # summary
    print(f"\n{'=' * 60}")
    print("  PIPELINE COMPLETE ✅")
    print("=" * 60)
    print(f"  Total time: {fmt_time(elapsed)}")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print("  Outputs:")
    for d, contents in [
        ("data/bronze/", "bronze_kaggle, bronze_rwc_matches, bronze_rankings, bronze_rwc_history"),
        ("data/silver/", "silver_results.parquet"),
        ("data/gold/",   "gold_train.parquet, gold_test.parquet"),
        ("models/",      "best_model.pkl"),
        ("figures/",     "PDF visualisations"),
        ("reports/",     "eda_summary, model_comparison, walk_forward_cv, rwc2027_results"),
    ]:
        print(f"  {d:<16} → {contents}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted — resume with: python3 run_pipeline.py --from <ID>")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
