"""
Generates a synthetic but realistic YouTube-comments dataset:
  - data/comments.csv        : labeled comments (text, sentiment_label, video_category, timestamp)
  - data/reference_stats.json: summary stats of the "training" distribution, used later
                                by src/monitor.py to detect drift in new comment batches

Why synthetic data? This project was built in a sandbox with no outbound access to the
real YouTube Data API (and no API key). See src/youtube_fetcher.py for the real
integration you can run yourself with a free Google Cloud API key — this file exists
so the rest of the pipeline (model, API, dashboard, monitoring) is fully testable
without one.
"""
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(7)

DATA_DIR = Path(__file__).parent
CATEGORIES = ["tutorial", "product_review", "vlog", "gaming", "music_video", "tech_news"]
TOPICS = ["the editing", "your explanation", "this tutorial", "the new update", "this build",
          "the audio mix", "your setup", "this series", "the intro", "the pacing"]
CREATORS_ASIDE = ["", " honestly", " ngl", " tbh", ""]

# Slot-filled templates (like the support-assistant project) so the dataset isn't
# just a handful of sentences repeated verbatim — that would make the classifier
# memorize instead of generalize, and inflate test accuracy artificially.
TEMPLATES = {
    "positive": [
        "This is exactly what I needed{aside}, thank you so much for explaining {topic} so clearly!",
        "Best video on this topic I've found, subscribed immediately.",
        "Wow, {topic} is next level{aside}, great work!",
        "I've watched this three times now, so well explained.",
        "This channel deserves way more views, underrated content.",
        "Finally a video that actually works{aside}, thank you!",
        "{topic} keeps getting better every episode, love it.",
        "This made my day, genuinely hilarious and well made.",
        "Really appreciate how much effort went into {topic}.",
        "You explained {topic} better than my actual course did.",
    ],
    "neutral": [
        "What software did you use for {topic}?",
        "Can you do a follow-up video on the advanced settings?",
        "What time does the next episode release?",
        "Is {topic} still relevant with the latest update?",
        "Does this work on the older model too?",
        "Where can I find the source code you mentioned?",
        "How long did {topic} take you to put together?",
        "Which microphone are you using for these videos?",
        "Do you have a video comparing this to the previous version?",
        "Is there a written version of {topic} somewhere?",
    ],
    "negative": [
        "The audio is way too quiet{aside}, had to max out my volume the whole video.",
        "This doesn't work at all, I followed every step and got an error.",
        "Way too many ads for a 5 minute video, unsubscribing.",
        "The thumbnail is misleading, this isn't what {topic} is about.",
        "Please stop dragging out {topic}, just get to the point.",
        "This is outdated information, none of this works anymore.",
        "Really disappointed{aside}, expected way more detail on {topic}.",
        "The click-bait title got me but {topic} doesn't deliver.",
        "{topic} is honestly kind of confusing, could use more detail.",
        "Not a fan of the recent changes to {topic}, felt rushed.",
    ],
}


def build_comments(n_per_class: int = 60) -> list:
    rows = []
    now = datetime.now()
    seen = set()
    for sentiment, sentences in TEMPLATES.items():
        count = 0
        attempts = 0
        while count < n_per_class and attempts < n_per_class * 20:
            attempts += 1
            sentence = random.choice(sentences)
            text = sentence.format(topic=random.choice(TOPICS), aside=random.choice(CREATORS_ASIDE))
            key = (sentiment, text)
            if key in seen:
                continue
            seen.add(key)
            count += 1
            days_ago = random.randint(0, 29)
            timestamp = (now - timedelta(days=days_ago, hours=random.randint(0, 23))).isoformat()
            rows.append({
                "text": text,
                "sentiment_label": sentiment,
                "video_category": random.choice(CATEGORIES),
                "timestamp": timestamp,
                "likes": random.randint(0, 500) if sentiment == "positive" else random.randint(0, 50),
            })
    random.shuffle(rows)
    return rows


def main():
    rows = build_comments()
    comments_path = DATA_DIR / "comments.csv"
    with open(comments_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "sentiment_label", "video_category", "timestamp", "likes"])
        writer.writeheader()
        writer.writerows(rows)

    # Reference stats used by src/monitor.py to flag drift in future comment batches
    lengths = [len(r["text"].split()) for r in rows]
    from collections import Counter
    sentiment_dist = Counter(r["sentiment_label"] for r in rows)
    total = len(rows)
    reference_stats = {
        "avg_comment_length_words": sum(lengths) / len(lengths),
        "sentiment_distribution": {k: round(v / total, 3) for k, v in sentiment_dist.items()},
        "n_reference_samples": total,
    }
    with open(DATA_DIR / "reference_stats.json", "w", encoding="utf-8") as f:
        json.dump(reference_stats, f, indent=2)

    print(f"Wrote {len(rows)} comments to {comments_path}")
    print("Reference stats:", reference_stats)


if __name__ == "__main__":
    main()
