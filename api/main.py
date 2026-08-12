"""
FastAPI service — this is the "production" surface of the project, separate from
the Streamlit dashboard. A Streamlit app is a demo; a REST API is what an actual
system would integrate against (e.g. a moderation tool calling /predict on new
comments as they arrive). Run:

    uvicorn api.main:app --reload --port 8000

Then see interactive docs at http://localhost:8000/docs (FastAPI generates this
automatically — worth mentioning in an interview, it's not extra work).
"""
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.sentiment_model import load_model, predict_sentiment
from src.retriever import load_index
from src.rag_qa import answer_question
from src.monitor import check_drift, log_prediction

ROOT = Path(__file__).parent.parent
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (ROOT / "models" / "sentiment_model.joblib").exists():
        raise RuntimeError("Model not found. Run `python train.py` before starting the API.")
    STATE["model"] = load_model()
    STATE["retriever"] = load_index()
    yield
    STATE.clear()


app = FastAPI(
    title="YouTube Sentiment MLOps API",
    description="Classifies YouTube comment sentiment and answers questions over a comment corpus via RAG.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This tutorial was so helpful, thank you!"])


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    all_scores: dict
    latency_ms: float


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What are viewers complaining about?"])
    top_k: int = 5
    sentiment_filter: str | None = None


class AskResponse(BaseModel):
    answer: str
    backend: str
    sources: list


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    result = predict_sentiment(STATE["model"], req.text)
    latency_ms = round((time.time() - start) * 1000, 2)

    log_prediction(req.text, result["sentiment"], result["confidence"])

    return PredictResponse(
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        latency_ms=latency_ms,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.sentiment_filter and req.sentiment_filter not in {"positive", "neutral", "negative"}:
        raise HTTPException(400, "sentiment_filter must be one of positive/neutral/negative")

    result = answer_question(
        req.question, STATE["retriever"], top_k=req.top_k, sentiment_filter=req.sentiment_filter
    )
    return AskResponse(**result)


@app.get("/monitor/drift")
def drift():
    """Compares recent logged predictions against the training-set reference
    distribution and flags if things have shifted (e.g. suddenly way more negative
    comments than the model was trained on) — the MLOps "is my model still valid"
    check."""
    return check_drift()
