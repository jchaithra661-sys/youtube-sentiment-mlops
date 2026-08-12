"""
Training entry point with MLflow experiment tracking — this is the core MLOps habit
this project demonstrates: every training run is logged (params, metrics, the model
artifact itself) instead of overwriting a single "final_model.pkl" and losing the
history of what was tried.

    python train.py                          # default hyperparameters
    python train.py --C 0.5 --ngram-max 1     # try a different run, logged separately
    mlflow ui                                 # inspect all runs in a browser at :5000

Also builds src/monitor.py's reference stats file so drift detection has a baseline.
"""
import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.sentiment_model import train_sentiment_model

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "comments.csv"
MLFLOW_DB = ROOT / "mlflow.db"
METRICS_PATH = ROOT / "models" / "current_metrics.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size for TF-IDF")
    args = parser.parse_args()

    if not DATA_CSV.exists():
        print("Data missing, generating synthetic dataset...")
        import subprocess
        subprocess.run(["python3", str(ROOT / "data" / "generate_data.py")], check=True)

    # SQLite-backed tracking store (MLflow's file store is now maintenance-mode-only
    # for new setups). This is also more realistic: a real MLOps setup almost always
    # points MLflow at a proper database, not loose files.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("youtube-sentiment")

    with mlflow.start_run():
        result = train_sentiment_model(str(DATA_CSV), C=args.C, ngram_max=args.ngram_max)

        mlflow.log_params(result["params"])
        mlflow.log_metric("accuracy", result["accuracy"])
        mlflow.log_metric("macro_f1", result["macro_f1"])
        mlflow.log_metric("n_train", result["n_train"])
        mlflow.log_metric("n_test", result["n_test"])
        mlflow.sklearn.log_model(result["pipeline"], name="model")

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump({"macro_f1": result["macro_f1"], "accuracy": result["accuracy"]}, f, indent=2)

        run = mlflow.active_run()
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Accuracy: {result['accuracy']:.3f} | Macro F1: {result['macro_f1']:.3f}\n")
        print(result["report"])

    print(f"\nModel saved to models/sentiment_model.joblib (used by the API/dashboard)")
    print(f"Full run history logged to {MLFLOW_DB} — run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse it")


if __name__ == "__main__":
    main()
