"""Text helpers shared by ingestion, retrieval and the heuristic reasoning provider."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """
    a an the and or but if then else of at by for with about against between into through during before after
    above below to from up down in out on off over under again further once here there when where why how all any
    both each few more most other some such no nor not only own same so than too very can will just don should now
    is are was were be been being have has had having do does did doing i me my myself we our ours ourselves you
    your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs
    themselves what which who whom this that these those am would could ought shall may might must also tell
    please give show list describe explain know find get let us something anything everything thing things
    """.split()
)

_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|[A-Za-z0-9]")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=\s*[-*•]\s)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalise unicode, strip control characters, de-hyphenate and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # words hyphenated across line breaks
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NL_RE.sub("\n\n", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        part = " ".join(part.split())
        part = re.sub(r"^[-*•]\s*", "", part)
        if part:
            sentences.append(part)
    return sentences


def words(text: str) -> list[str]:
    return [w.lower().strip(".") for w in _WORD_RE.findall(text)]


def _stem(word: str) -> str:
    for suffix in ("ies", "es", "s"):
        if len(word) > 4 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def content_terms(text: str) -> list[str]:
    """Lower-cased, lightly stemmed, stop-word-free terms (order preserved, duplicates kept)."""
    return [_stem(w) for w in words(text) if w not in STOPWORDS and len(w) > 1]


def term_set(text: str) -> set[str]:
    return set(content_terms(text))


def overlap_ratio(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & term_set(text)) / len(query_terms)


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_NORMALIZE_STRIP = re.compile(r"[^a-z0-9+#]+")


def normalize_name(name: str) -> str:
    """Canonical key for entity names: casefold, strip punctuation, collapse spaces."""
    name = unicodedata.normalize("NFKC", name).casefold()
    name = _NORMALIZE_STRIP.sub(" ", name)
    return " ".join(name.split())


def safe_filename(filename: str, max_length: int = 200) -> str:
    """Strip path components and unsafe characters from a client-supplied filename."""
    filename = filename.replace("\\", "/").split("/")[-1]
    filename = unicodedata.normalize("NFKC", filename)
    filename = re.sub(r"[^A-Za-z0-9._ \-()]", "_", filename).strip(" .")
    if not filename:
        filename = "document"
    if len(filename) > max_length:
        stem, dot, ext = filename.rpartition(".")
        filename = (stem[: max_length - len(ext) - 1] + dot + ext) if dot else filename[:max_length]
    return filename
