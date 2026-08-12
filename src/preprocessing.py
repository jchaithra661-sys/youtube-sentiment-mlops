"""
Text cleaning for YouTube comments. Real comments are messier than the support
tickets in the other project — emojis, @mentions, timestamps ("2:34 lol"), links,
ALL CAPS, repeated punctuation — so this module exists separately instead of being
folded into the model file.
"""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
REPEATED_PUNCT_RE = re.compile(r"([!?.]){2,}")
MULTI_SPACE_RE = re.compile(r"\s{2,}")

# Common emoji ranges — stripped rather than kept, since the TF-IDF model downstream
# doesn't use them as tokens. (If you want emoji-aware sentiment, that's a documented
# "Level up" in the README — emojis are often a *strong* sentiment signal on YouTube.)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)


def clean_comment(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = TIMESTAMP_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)  # "!!!" -> "!"
    text = MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_batch(texts: list) -> list:
    return [clean_comment(t) for t in texts]


if __name__ == "__main__":
    examples = [
        "This is AMAZING!!! check out my channel www.example.com @someuser",
        "lol at 2:34 this is so relatable \U0001F602\U0001F602",
        "  Too    many    spaces   here   ",
    ]
    for ex in examples:
        print(repr(ex), "->", repr(clean_comment(ex)))
