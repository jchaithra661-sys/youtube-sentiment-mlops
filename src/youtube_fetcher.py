"""
Real YouTube Data API v3 integration — pulls actual comments from a real video.

This needs a free API key (YouTube Data API v3, enabled in Google Cloud Console —
takes about 5 minutes to set up: console.cloud.google.com > APIs & Services >
enable "YouTube Data API v3" > Credentials > Create API key).

Set it as an environment variable before running:
    export YOUTUBE_API_KEY="your-key-here"
    python src/youtube_fetcher.py --video-id dQw4w9WgXcQ --max-results 200

This module was written but NOT run against the live API while building this project
(no API key / outbound access in the build environment) — the rest of the pipeline
(model, API, dashboard, monitoring) runs entirely on data/comments.csv instead so it's
testable without one. Running this file yourself with your own key is the natural
"make it real" step — see the project guide.
"""
import argparse
import csv
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def fetch_comments(video_id: str, api_key: str, max_results: int = 200) -> list:
    """Fetch top-level comments for a video via the YouTube Data API v3.
    Returns a list of dicts matching the schema of data/comments.csv (minus the
    sentiment_label column, since real comments obviously aren't pre-labeled)."""
    from googleapiclient.discovery import build  # google-api-python-client

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet["textDisplay"],
                "video_category": None,  # not provided per-comment by the API; tag manually or via video metadata
                "timestamp": snippet["publishedAt"],
                "likes": snippet.get("likeCount", 0),
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments[:max_results]


def save_raw_comments(comments: list, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "video_category", "timestamp", "likes"])
        writer.writeheader()
        writer.writerows(comments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch real YouTube comments for a video")
    parser.add_argument("--video-id", required=True, help="YouTube video ID (the v= part of the URL)")
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--out", default=str(DATA_DIR / "raw_youtube_comments.csv"))
    args = parser.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set YOUTUBE_API_KEY first. See this file's docstring for how to get one (free, ~5 min)."
        )

    comments = fetch_comments(args.video_id, api_key, args.max_results)
    save_raw_comments(comments, Path(args.out))
    print(f"Fetched {len(comments)} comments -> {args.out}")
    print("Next: run these through src/preprocessing.py and src/sentiment_model.py to classify them.")
