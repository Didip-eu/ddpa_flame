"""The display tokens of FLAME's reports: runs of word characters and single other characters, each an
exact substring of the text, so the reports and the spans point at the text as written."""

import re
import unicodedata

TOKEN = re.compile(r"\w+|[^\w\s]")
# No space goes before these, nor after OPENING.
CLOSING = frozenset(".,;:!?)]}%'\"")
OPENING = frozenset("([{")


def tokenize(text: str) -> list[str]:
    """The tokens of a text, in order."""
    return TOKEN.findall(text)


def _attaches(previous: str, token: str) -> bool:
    return token[0] in CLOSING or previous[-1] in OPENING or unicodedata.combining(token[0]) > 0


def detokenize(tokens: list[str]) -> str:
    """Tokens as running text: a space between two, except before closing punctuation and combining
    marks and after an opening bracket."""
    if not tokens:
        return ""
    parts = [tokens[0]]
    for previous, token in zip(tokens, tokens[1:]):
        parts.append(token if _attaches(previous, token) else " " + token)
    return "".join(parts)
