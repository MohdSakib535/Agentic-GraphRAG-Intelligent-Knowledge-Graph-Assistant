"""Token counting. Uses tiktoken when its encoding is available, otherwise a
calibrated regex approximation (works offline / behind restrictive proxies)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_APPROX_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@lru_cache(maxsize=1)
def _encoding() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _approx_tokens(text: str) -> list[str]:
    pieces: list[str] = []
    for piece in _APPROX_RE.findall(text):
        # Long words are split into ~4-char sub-tokens, mimicking BPE behaviour.
        if len(piece) > 8:
            pieces.extend(piece[i : i + 4] for i in range(0, len(piece), 4))
        else:
            pieces.append(piece)
    return pieces


def count_tokens(text: str) -> int:
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return len(_approx_tokens(text))


def tokenizer_name() -> str:
    return "tiktoken:cl100k_base" if _encoding() is not None else "regex-approx"
