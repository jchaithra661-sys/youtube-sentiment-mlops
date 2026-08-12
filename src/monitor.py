"""
The monitoring layer — the part of MLOps that's easy to skip in a portfolio project
and exactly why including it stands out. Two things:

1. log_prediction() appends every prediction the API serves to a local log
   (models/prediction_log.csv), the way you'd ship logs to a monitoring system in
   production.
2. check_drift() compares the sentiment distribution of recent predictions against
   the training-set reference distribution (data/reference_stats.json) using a
   chi-squared test, and flags it if they've diverged — e.g. if a model trained
   mostly on neutral/positive comments suddenly sees a wave of negative ones, that's
   a signal worth a human looking at, whether it's real user sentiment shifting or
   the model degrading.
"""
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from scipy.stats import chisquare

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "models" / "prediction_log.csv"
REFERENCE_STATS_PATH = ROOT / "data" / "reference_stats.json"

LABELS = ["positive", "neutral", "negative"]


def log_prediction(text: str, sentiment: str, confidence: float):
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "text", "sentiment", "confidence"])
        writer.writerow([datetime.now().isoformat(), text, sentiment, confidence])


def _load_reference() -> dict:
    with open(REFERENCE_STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def check_drift(min_samples: int = 20) -> dict:
    if not LOG_PATH.exists():
        return {"status": "no_data", "message": "No predictions logged yet."}

    with open(LOG_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(rows)} predictions logged, need {min_samples} for a reliable drift check.",
            "n_logged": len(rows),
        }

    reference = _load_reference()
    ref_dist = reference["sentiment_distribution"]

    recent = rows[-200:]  # sliding window: only check drift on the most recent batch
    observed_counts = Counter(r["sentiment"] for r in recent)
    n = len(recent)

    observed = [observed_counts.get(label, 0) for label in LABELS]
    raw_expected = [ref_dist.get(label, 0) * n for label in LABELS]
    raw_expected = [max(e, 1e-6) for e in raw_expected]  # chisquare needs non-zero expected counts
    # ref_dist values are stored rounded to 3 decimals (e.g. 0.361+0.361+0.277 = 0.999,
    # not exactly 1.0), so raw_expected can be off from `n` by a tiny amount. scipy's
    # chisquare requires sum(observed) == sum(expected) exactly (within ~1e-8), so
    # rescale to force that — this doesn't change the *shape* of the expected
    # distribution, only corrects the rounding-induced total.
    scale = n / sum(raw_expected)
    expected = [e * scale for e in raw_expected]

    stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    drifted = p_value < 0.05

    return {
        "status": "drift_detected" if drifted else "stable",
        "p_value": round(float(p_value), 4),
        "chi2_statistic": round(float(stat), 4),
        "n_recent_samples": n,
        "observed_distribution": {l: round(c / n, 3) for l, c in zip(LABELS, observed)},
        "reference_distribution": ref_dist,
        "interpretation": (
            "The recent sentiment mix differs significantly from training data — "
            "investigate (real shift in audience sentiment, or the model may need retraining)."
            if drifted else
            "Recent predictions are statistically consistent with the training distribution."
        ),
    }


if __name__ == "__main__":
    # simulate some logged predictions for a manual smoke test
    import random
    random.seed(1)
    for _ in range(30):
        log_prediction("sample comment", random.choice(LABELS), round(random.uniform(0.5, 0.99), 2))
    print(check_drift())
