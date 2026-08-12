"""
Embedding + retrieval layer for "ask questions about your comments" (RAG over the
comment corpus, rather than over a static FAQ like the support-assistant project).

Embeddings here are TF-IDF -> TruncatedSVD (LSA), the same technique used in the
support-assistant project, for the same reason: it's a real, well-established dense
embedding method that runs fully offline with no GPU and no multi-hundred-MB model
download (a `sentence-transformers` install was attempted while building this and
pulled in a 500MB+ torch/CUDA wheel that the sandbox's network couldn't complete —
see README "Level up" for the exact one-function swap to use it on your own machine).

The interface (`build_index` / `retrieve`) is intentionally identical to what you'd
write for neural embeddings + FAISS, so upgrading later doesn't touch any calling code.
"""
import json
import joblib
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.preprocessing import clean_batch

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
VECTORIZER_PATH = MODELS_DIR / "retriever_tfidf.joblib"
SVD_PATH = MODELS_DIR / "retriever_svd.joblib"
INDEX_PATH = MODELS_DIR / "retriever.faiss"
DOCS_PATH = MODELS_DIR / "retriever_docs.json"


def build_index(csv_path: str, n_components: int = 80, save: bool = True) -> dict:
    df = pd.read_csv(csv_path)
    clean_texts = clean_batch(df["text"].tolist())

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(clean_texts)

    n_comp = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    dense = svd.fit_transform(tfidf_matrix).astype("float32")

    faiss.normalize_L2(dense)
    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    docs = df.to_dict(orient="records")

    if save:
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(svd, SVD_PATH)
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f)

    return {"vectorizer": vectorizer, "svd": svd, "index": index, "docs": docs}


def load_index() -> dict:
    vectorizer = joblib.load(VECTORIZER_PATH)
    svd = joblib.load(SVD_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)
    return {"vectorizer": vectorizer, "svd": svd, "index": index, "docs": docs}


def retrieve(query: str, resources: dict, top_k: int = 5, sentiment_filter: str = None) -> list:
    vectorizer, svd, index, docs = (
        resources["vectorizer"], resources["svd"], resources["index"], resources["docs"]
    )
    clean_q = clean_batch([query])[0]
    q_tfidf = vectorizer.transform([clean_q])
    q_dense = svd.transform(q_tfidf).astype("float32")
    faiss.normalize_L2(q_dense)

    # over-fetch then filter by sentiment if requested, so filtering doesn't starve top_k
    fetch_k = top_k * 4 if sentiment_filter else top_k
    scores, idxs = index.search(q_dense, min(fetch_k, len(docs)))

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        doc = docs[idx]
        if sentiment_filter and doc.get("sentiment_label") != sentiment_filter:
            continue
        results.append({**doc, "similarity": round(float(score), 3)})
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    csv_path = str(Path(__file__).parent.parent / "data" / "comments.csv")
    resources = build_index(csv_path)
    for q in ["complaints about audio quality", "people asking about editing software", "positive feedback on tutorials"]:
        print(f"\nQuery: {q}")
        for r in retrieve(q, resources, top_k=3):
            print(f"  [{r['similarity']}] ({r['sentiment_label']}) {r['text']}")
