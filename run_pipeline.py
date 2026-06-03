import subprocess
import sys
from pathlib import Path

PYTHON = str(Path(__file__).parent / ".venv" / "bin" / "python3")

steps = [
    "scripts/00_data_ingestion.py",
    "scripts/01_data_cleaning.py",
    "scripts/02_features_gold.py",
    "scripts/03_analysis_eda.py",
    "scripts/04_model_logistic_regression.py",
    "scripts/05_model_xgboost_comparison.py",
    "scripts/06_tier_analysis.py",
]

for step in steps:
    print(f"--- Running {step} ---")
    result = subprocess.run([PYTHON, step])
    if result.returncode != 0:
        print(f"Error in {step}, stopping.")
        sys.exit(1)

print("Pipeline complete.")
