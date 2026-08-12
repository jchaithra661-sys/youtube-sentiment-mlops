# YouTube Viewer Sentiment — MLOps Project

Classifies YouTube comment sentiment, answers questions over the comment corpus using retrieval-augmented generation (RAG), and wraps the whole thing in an actual MLOps pipeline: experiment tracking, a served API, automated tests in CI, drift monitoring, and gated automatic retraining.

This project pairs with a separate RAG project I built first (an AI customer support assistant). That one proved I can build the ML/NLP pipeline; this one proves I can operate it — the two together cover both halves of what "AI & Automation" roles are actually asking for.

## Architecture

```
YouTube comments (real, via YouTube Data API — or the bundled synthetic dataset)
      │
      ├──► Preprocessing      (strip emojis, links, @mentions, timestamps)
      │
      ├──► Sentiment model    (TF-IDF + Logistic Regression, 3-class)
      │         │
      │         └──► MLflow experiment tracking (every training run logged)
      │
      └──► Retriever          (TF-IDF → LSA embeddings → FAISS) + RAG Q&A
                │
                ▼
      ┌─────────┴─────────┐
      │                   │
 FastAPI service    Streamlit dashboard
 (/predict, /ask,   (trends, live classify,
  /monitor/drift)    ask-a-question, drift check)
      │
      ▼
 prediction_log.csv ──► drift check (chi-squared test vs. training distribution)
      │
      ▼
 scripts/retrain.py ──► scheduled retrain, only promotes if macro-F1 doesn't regress
```

## Results

- Sentiment classifier: **91.2% accuracy, 0.915 macro F1** on a held-out test split across 3 balanced classes (166 comments — see `data/generate_data.py` for how and why).
- Retrieval: FAISS cosine similarity search over LSA embeddings, with a sentiment filter for "show me only negative comments about X" style queries.
- **13/13 automated tests passing** (`pytest tests/ -v`) — 7 on the ML pipeline, 6 on the API contract (via FastAPI's TestClient, no server process needed — this is what runs in CI).
- CI runs the entire pipeline (generate data → train → build index → test) on every push — see `.github/workflows/ci.yml`.

## Quickstart

```bash
pip install -r requirements.txt
python train.py                                    # trains model, logs to MLflow
python -c "from src.retriever import build_index; build_index('data/comments.csv')"

uvicorn api.main:app --reload --port 8000           # API: http://localhost:8000/docs
streamlit run dashboard/app.py                      # Dashboard: http://localhost:8501
```

Run the tests:
```bash
python -m pytest tests/ -v
```

Inspect training run history:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

Run everything in Docker instead:
```bash
docker compose up --build
```

## Project layout

```
youtube-sentiment-mlops/
├── data/
│   ├── generate_data.py       # synthetic dataset (see note below)
│   ├── comments.csv           # 166 labeled comments
│   └── reference_stats.json   # training-distribution baseline, used by monitor.py
├── src/
│   ├── youtube_fetcher.py     # REAL YouTube Data API integration (needs your own key)
│   ├── preprocessing.py       # comment text cleaning
│   ├── sentiment_model.py     # TF-IDF + Logistic Regression
│   ├── retriever.py           # TF-IDF → LSA → FAISS retrieval
│   ├── rag_qa.py               # RAG: retrieve comments + synthesize an answer
│   └── monitor.py             # prediction logging + drift detection (chi-squared)
├── api/main.py                 # FastAPI service: /predict, /ask, /monitor/drift
├── dashboard/app.py            # Streamlit dashboard (4 tabs)
├── train.py                    # trains model, logs to MLflow, builds current_metrics.json
├── scripts/retrain.py          # gated retraining: only promotes if not worse
├── .github/workflows/
│   ├── ci.yml                  # tests on every push
│   └── scheduled_retrain.yml   # weekly retrain + auto-commit if promoted
├── Dockerfile, Dockerfile.dashboard, docker-compose.yml
└── tests/
    ├── test_pipeline.py        # 7 tests: preprocessing, model, retriever, RAG
    └── test_api.py             # 6 tests: API contract via FastAPI TestClient
```

## A note on the dataset, and on embeddings

Like the support-assistant project, `data/comments.csv` is synthetic — hand-authored with intentional variation (not just repeated sentences, which would let the classifier memorize instead of generalize) so it runs offline and reproducibly. `src/youtube_fetcher.py` is the real integration: point it at any video with a free YouTube Data API key and it pulls actual comments in the same schema.

The retriever uses TF-IDF + TruncatedSVD (LSA) instead of neural embeddings (`sentence-transformers`). This was a deliberate choice under a real constraint: installing `sentence-transformers` in the build environment pulled in a 500MB+ torch/CUDA wheel that the sandbox's network couldn't complete downloading. LSA is a legitimate, well-established dense embedding technique — not a placeholder — and `src/retriever.py`'s docstring explains exactly how to swap in `sentence-transformers` with the FAISS indexing code completely unchanged.

## Known limitations

- LSA embeddings are weaker than neural embeddings on abstract/paraphrased queries (e.g. "what are people upset about" doesn't lexically overlap with "the audio is way too quiet" as well as a real sentence embedding model would). The `sentiment_filter` parameter on `retrieve()`/`answer_question()` is the practical workaround — combine keyword search with a hard sentiment filter rather than relying on semantic search alone.
- No true streaming ingestion — comments are batch-loaded, not consumed as they arrive.
- The retrain promotion gate compares only macro-F1; a real system would also check for per-class regressions and require a minimum sample size before promoting.

## Level up (optional extensions)

1. **Real embeddings**: swap `TfidfVectorizer + TruncatedSVD` in `src/retriever.py` for `SentenceTransformer("all-MiniLM-L6-v2").encode(...)` — same FAISS code below it.
2. **Real YouTube data**: get a free API key, run `python src/youtube_fetcher.py --video-id <id>`, then label a sample and retrain.
3. **Live LLM generation**: `pip install openai`, set `OPENAI_API_KEY` — `rag_qa.py` switches from template to real generation automatically.
4. **True streaming**: replace the batch CSV load with a queue (even a simple polling loop against the YouTube API on a schedule) feeding `/predict`.
5. **Deploy**: push the Docker images somewhere free (Render, Fly.io) so `/docs` and the dashboard are live links, not just local instructions.

## License

MIT — see `LICENSE`.
