"""
Automated retraining with a promotion gate — the MLOps idea that a retrain should
never silently replace a working model with a worse one.

    python scripts/retrain.py

What it does:
1. Trains a new model (logged as a new MLflow run, same as train.py).
2. Compares its macro-F1 against the currently-deployed model's last known score
   (models/current_metrics.json).
3. Only overwrites models/sentiment_model.joblib if the new one is at least as good;
   otherwise it keeps the old model and exits with a warning.

In a real setup this would be triggered on a schedule (a scheduled GitHub Actions
workflow, a cron job, or an Airflow DAG) whenever new labeled data comes in — wire
that up as your own "Level up" once you're comfortable with what this script does.
"""
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sentiment_model import train_sentiment_model

ROOT = Path(__file__).parent.parent
DATA_CSV = ROOT / "data" / "comments.csv"
METRICS_PATH = ROOT / "models" / "current_metrics.json"
MLFLOW_DB = ROOT / "mlflow.db"


def load_current_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"macro_f1": 0.0}


def main():
    current = load_current_metrics()
    print(f"Currently deployed model macro F1: {current['macro_f1']:.3f}")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("youtube-sentiment")

    with mlflow.start_run(run_name="scheduled_retrain"):
        result = train_sentiment_model(str(DATA_CSV), save=False)  # don't overwrite yet
        mlflow.log_metric("accuracy", result["accuracy"])
        mlflow.log_metric("macro_f1", result["macro_f1"])

        print(f"New candidate model macro F1: {result['macro_f1']:.3f}")

        if result["macro_f1"] >= current["macro_f1"]:
            import joblib
            joblib.dump(result["pipeline"], ROOT / "models" / "sentiment_model.joblib")
            with open(METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump({"macro_f1": result["macro_f1"], "accuracy": result["accuracy"]}, f, indent=2)
            mlflow.log_param("promoted", True)
            print("New model PROMOTED — deployed to models/sentiment_model.joblib")
        else:
            mlflow.log_param("promoted", False)
            print("New model NOT promoted (worse than current) — keeping existing model")


if __name__ == "__main__":
    main()
