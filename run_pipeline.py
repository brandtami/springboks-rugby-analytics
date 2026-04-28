import subprocess

steps = [
    "scripts/00_data_ingestion.py",
    "scripts/01_data_cleaning.py",
    "scripts/02_features_gold.py",
    "scripts/03_analysis_eda.py",
    "scripts/04_model_logistic_regression.py",
    "scripts/05_model_xgboost_comparison.py",
]

for step in steps:
    print(f"\n--- Running {step} ---")
    result = subprocess.run(["python", step])

    if result.returncode != 0:
        print(f"Error in {step}")
        break