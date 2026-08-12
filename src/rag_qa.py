"""
RAG question-answering over the comment corpus: "What are viewers unhappy about?" /
"What do people ask about the editing?" — retrieves the most relevant real comments
(src/retriever.py) and synthesizes an answer from them, instead of an LLM guessing
from its own training data.

Same two-backend design as the support-assistant project's generator.py: template
mode by default (free, offline, always available for a demo), live LLM mode if
OPENAI_API_KEY is set. Kept as a second, separate module (not reused from the other
project) because the prompt and the summarization logic are genuinely different here
— summarizing N noisy comments into a trend is a different task from answering one
FAQ lookup.
"""
import os
from collections import Counter

from src.retriever import retrieve

SYSTEM_PROMPT = (
    "You are analyzing real YouTube comments for a content creator. Given a question "
    "and a set of retrieved comments, write a concise (under 100 words) summary that "
    "directly answers the question, grounded ONLY in the comments provided. Mention "
    "roughly how many comments support your summary. Do not invent comments that "
    "weren't given to you."
)


def _llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _answer_with_llm(question: str, comments: list) -> str:
    from openai import OpenAI

    client = OpenAI()
    context = "\n".join(f"- [{c['sentiment_label']}] {c['text']}" for c in comments)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nComments:\n{context}"},
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _answer_with_template(question: str, comments: list) -> str:
    if not comments:
        return "I couldn't find any comments closely related to that question."

    sentiment_counts = Counter(c["sentiment_label"] for c in comments)
    total = len(comments)
    dominant = sentiment_counts.most_common(1)[0][0]

    summary = (
        f"Out of {total} closely related comments, most are {dominant} "
        f"({dict(sentiment_counts)}). "
    )
    example = comments[0]["text"]
    summary += f'A representative comment: "{example}"'
    return summary


def answer_question(question: str, resources: dict, top_k: int = 5, sentiment_filter: str = None) -> dict:
    comments = retrieve(question, resources, top_k=top_k, sentiment_filter=sentiment_filter)

    if _llm_available():
        try:
            text = _answer_with_llm(question, comments)
            return {"answer": text, "backend": "openai:gpt-4o-mini", "sources": comments}
        except Exception as exc:
            fallback = _answer_with_template(question, comments)
            return {"answer": fallback, "backend": f"template (LLM call failed: {exc})", "sources": comments}

    text = _answer_with_template(question, comments)
    return {"answer": text, "backend": "template", "sources": comments}


if __name__ == "__main__":
    from src.retriever import build_index

    resources = build_index(str("data/comments.csv"), save=False)
    for q in ["What are viewers complaining about?", "What do people like about the videos?"]:
        result = answer_question(q, resources, top_k=5)
        print(f"\nQ: {q}")
        print(f"A [{result['backend']}]: {result['answer']}")
