"""
Sanity tests for the core ML pipeline (model + retriever + RAG QA).
API-level tests live in test_api.py — kept separate since they test a different
layer (HTTP contracts vs. model behavior).

Run:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.preprocessing import clean_comment
from src.sentiment_model import load_model, predict_sentiment
from src.retriever import load_index, retrieve
from src.rag_qa import answer_question


def test_preprocessing_strips_noise():
    dirty = "Check this out!!! www.example.com @someone lol at 2:34 \U0001F602"
    clean = clean_comment(dirty)
    assert "http" not in clean
    assert "@someone" not in clean
    assert "2:34" not in clean
    assert "!!!" not in clean


def test_sentiment_model_loads_and_predicts():
    model = load_model()
    result = predict_sentiment(model, "This tutorial was so helpful, thank you!")
    assert result["sentiment"] in {"positive", "neutral", "negative"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["all_scores"].values()) - 1.0) < 0.01


def test_sentiment_model_predicts_negative_correctly():
    model = load_model()
    result = predict_sentiment(model, "This is terrible, the audio is unusable and support ignored me")
    assert result["sentiment"] == "negative"


def test_retriever_returns_ordered_results():
    resources = load_index()
    results = retrieve("editing software question", resources, top_k=5)
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_retriever_sentiment_filter_only_returns_matching_sentiment():
    resources = load_index()
    results = retrieve("audio quality", resources, top_k=5, sentiment_filter="negative")
    assert all(r["sentiment_label"] == "negative" for r in results)


def test_rag_qa_smoke():
    resources = load_index()
    result = answer_question("What are viewers complaining about?", resources, top_k=5)
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0
    assert result["backend"] in {"template"} or result["backend"].startswith("openai") or "template" in result["backend"]


def test_rag_qa_handles_no_matches_gracefully():
    resources = load_index()
    result = answer_question("asdkjaslkdj qwoeiqwoei", resources, top_k=5, sentiment_filter="negative")
    assert isinstance(result["answer"], str)
