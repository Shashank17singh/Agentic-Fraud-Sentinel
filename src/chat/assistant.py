import json
import logging
import os
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")

# Below this cosine similarity the best passage does not really answer the
# question, so the assistant asks for clarification instead of guessing.
# shortcut: fixed by hand on a few sample questions; off-topic questions sharing a
# word like "best" can still pass. Groq mode screens those out; tune on a labelled set.
MIN_SCORE = 0.12
MIN_WORDS = 3
TOP_K = 2

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Groq sits behind Cloudflare, which rejects the default Python-urllib agent (1010).
USER_AGENT = "agentic-fraud-sentinel/1.0"
NO_ANSWER = "NO_ANSWER"

logger = logging.getLogger(__name__)

with open(KNOWLEDGE_PATH, encoding="utf-8") as f:
    PASSAGES = json.load(f)

_vectorizer = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), sublinear_tf=True
)
_matrix = _vectorizer.fit_transform(f"{p['title']} {p['text']}" for p in PASSAGES)

CLARIFY = (
    "I could not find a source in my collection that answers this. I can help with "
    "credit card fraud (lost cards, phishing, skimming, reporting fraud) and with "
    "checking whether a news story is reliable. Could you rephrase or add detail?"
)


def retrieve(question: str, top_k: int = TOP_K) -> list[tuple[dict, float]]:
    """Return the best-matching passages and their similarity, best first."""
    scores = (_matrix @ _vectorizer.transform([question]).T).toarray().ravel()
    best = np.argsort(scores)[::-1][:top_k]
    return [(PASSAGES[i], float(scores[i])) for i in best if scores[i] >= MIN_SCORE]


def _cite(hits: list[tuple[dict, float]]) -> list[tuple[int, dict, float]]:
    """Number hits by source page, so passages from the same page share one [n]."""
    numbers: dict[str, int] = {}
    return [(numbers.setdefault(p["url"], len(numbers) + 1), p, s) for p, s in hits]


def _groq_reply(question: str, cited: list[tuple[int, dict, float]], key: str) -> str:
    """Ask Groq to answer only from the numbered passages, citing them as [n]."""
    context = "\n\n".join(f"[{n}] {p['text']}" for n, p, _ in cited)
    body = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer in two to four plain sentences using only the numbered "
                    "passages. Cite each claim with its passage number like [1]. If "
                    f"the passages do not answer the question, reply {NO_ANSWER} only."
                ),
            },
            {
                "role": "user",
                "content": f"Passages:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())["choices"][0]["message"]["content"].strip()


def _clarify(mode: str) -> dict:
    return {
        "answer": CLARIFY,
        "sources": [],
        "topic": None,
        "grounded": False,
        "mode": mode,
    }


def answer(question: str) -> dict:
    """
    Answer a question about credit card fraud or fake news from cited passages.

    Returns the reply text with [n] markers, the cited sources, the topic of the
    best match, whether the reply is grounded in a source, and its mode: "groq"
    when Groq wrote it from the passages (needs GROQ_API_KEY), else "retrieval".
    """
    question = question.strip()
    hits = retrieve(question) if len(question.split()) >= MIN_WORDS else []
    if not hits:
        return _clarify("retrieval")

    cited = _cite(hits)
    # hits are best first, so each page keeps the score of its best passage
    sources = []
    for n, p, score in cited:
        if n > len(sources):
            sources.append(
                {"n": n, "title": p["title"], "url": p["url"], "score": round(score, 3)}
            )
    reply = "\n\n".join(f"{p['text']} [{n}]" for n, p, _ in cited)
    mode = "retrieval"
    key = os.environ.get("GROQ_API_KEY")
    if key:
        try:
            reply, mode = _groq_reply(question, cited, key), "groq"
            if NO_ANSWER in reply:
                return _clarify(mode)
        # network/HTTP errors are OSError; bad JSON or shape is ValueError/KeyError/IndexError
        except (OSError, ValueError, KeyError, IndexError) as exc:
            logger.warning("Groq call failed, replying with passages: %s", exc)
    return {
        "answer": reply,
        "sources": sources,
        "topic": hits[0][0]["topic"],
        "grounded": True,
        "mode": mode,
    }
