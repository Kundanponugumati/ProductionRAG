"""Minimal text normalisation. Deliberately conservative: never drop content."""

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "opera-\ntions" → "operations"
_INLINE_SPACE = re.compile(r"[ \t]+")  # runs of spaces/tabs (NFKC already turned nbsp into spaces)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # ligatures (ﬁ → fi), full-width chars
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = (_INLINE_SPACE.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)
