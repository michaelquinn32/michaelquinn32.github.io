#!/usr/bin/env python
"""Deterministic writing statistics for a Markdown draft.

Signal, not a gate. Complements Vale by computing what Vale's `metric` extension
cannot: the distribution of sentence lengths (mean, standard deviation, min/max,
and how many sentences run long). Michael's voice guide prizes "burstiness,"
short punches alternating with long explanatory sentences, so a low standard
deviation is a real style signal (and a mild AI tell). Also reports a couple of
readability grades for trend.

Pure stdlib, offline, no ML. Usage:

    python writing_stats.py <file.md>

Prints a JSON object to stdout.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys


def _strip_frontmatter_and_code(text: str) -> str:
    """Drop YAML frontmatter and fenced code blocks; they are not prose."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Drop Markdown headings, list markers, and link URLs (keep link text).
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`[^`]+`", " ", text)
    return text


_SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s+|$)")
_WORD = re.compile(r"[A-Za-z0-9']+")
_VOWEL_RUN = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _syllables(word: str) -> int:
    """Rough syllable count (heuristic, good enough for grade-level trend)."""
    w = word.lower().strip("'")
    if not w:
        return 0
    count = len(_VOWEL_RUN.findall(w))
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(json.dumps({"status": "error", "message": "usage: writing_stats.py <file.md>"}))
        return 2
    path = pathlib.Path(argv[1])
    if not path.is_file():
        print(json.dumps({"status": "error", "message": f"not a file: {path}"}))
        return 2

    prose = _strip_frontmatter_and_code(path.read_text(encoding="utf-8"))

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(prose) if s.strip()]
    lengths = [len(_WORD.findall(s)) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    words = sum(lengths)
    syllables = sum(_syllables(w) for s in sentences for w in _WORD.findall(s))
    n_sent = len(lengths)

    if n_sent == 0 or words == 0:
        print(json.dumps({"file": str(path), "status": "empty"}))
        return 0

    mean_len = words / n_sent
    stdev_len = statistics.pstdev(lengths) if n_sent > 1 else 0.0
    long_threshold = 40
    short_threshold = 8

    flesch_kincaid = 0.39 * (words / n_sent) + 11.8 * (syllables / words) - 15.59
    reading_ease = 206.835 - 1.015 * (words / n_sent) - 84.6 * (syllables / words)

    result = {
        "file": str(path),
        "status": "ok",
        "sentences": n_sent,
        "words": words,
        "sentence_length": {
            "mean": round(mean_len, 1),
            "stdev": round(stdev_len, 1),
            "min": min(lengths),
            "max": max(lengths),
            "long_count": sum(1 for n in lengths if n > long_threshold),
            "short_count": sum(1 for n in lengths if n <= short_threshold),
        },
        "readability": {
            "flesch_kincaid_grade": round(flesch_kincaid, 1),
            "flesch_reading_ease": round(reading_ease, 1),
        },
        "notes": (
            "Signal, not a gate. A healthy long-form voice mixes lengths: expect a "
            "stdev in double digits and a supply of both short (<=8) and long (>40) "
            "sentences. Low stdev with a high mean reads as uniformly dense."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
