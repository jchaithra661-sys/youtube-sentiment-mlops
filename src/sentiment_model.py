"""
3-class sentiment classifier (positive / neutral / negative) for YouTube comments:
TF-IDF + Logistic Regression, same reasoning as the intent classifier in the support
assistant project — a few hundred labeled comments is not enough to fine-tune a
transformer without overfitting; a strong linear baseline is the correct call, and
every MLflow run logs the choice, hyperparameters, and metrics so it's an auditable
decision, not a guess.
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_batch

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"


def load_comments(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["clean_text"] = clean_batch(df["text"].tolist())
    return df


def train_sentiment_model(csv_path: str, C: float = 1.0, ngram_max: int = 2, save: bool = True) -> dict:
    df = load_comments(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment_label"], test_size=0.2, random_state=42,
        stratify=df["sentiment_label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, ngram_max), min_df=1, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=C, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, zero_division=0)

    if save:
        joblib.dump(pipeline, MODEL_PATH)

    return {
        "pipeline": pipeline, "accuracy": accuracy, "macro_f1": macro_f1,
        "report": report, "params": {"C": C, "ngram_max": ngram_max},
        "n_train": len(X_train), "n_test": len(X_test),
    }


def load_model(path: str = None):
    return joblib.load(path or MODEL_PATH)


def predict_sentiment(pipeline, text: str) -> dict:
    clean = clean_batch([text])[0]
    pred = pipeline.predict([clean])[0]
    proba = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    return {
        "sentiment": pred,
        "confidence": round(float(max(proba)), 3),
        "all_scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


if __name__ == "__main__":
    result = train_sentiment_model(str(Path(__file__).parent.parent / "data" / "comments.csv"))
    print(f"Accuracy: {result['accuracy']:.3f} | Macro F1: {result['macro_f1']:.3f}\n")
    print(result["report"])
