"""The display tokens of FLAME's reports: runs of word characters and single other characters, each an
exact substring of the text, so the reports and the spans point at the text as written."""

import re
import unicodedata

TOKEN = re.compile(r"\w+|[^\w\s]")
# No space goes before CLOSING, after OPENING, or on either side of JOINING; double quotes
# alternate between opening and closing.
CLOSING = frozenset(".,;:!?)]}%'")
OPENING = frozenset("([{")
JOINING = frozenset("/")


def tokenize(text: str) -> list[str]:
    """The tokens of a text, in order."""
    return TOKEN.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Tokens as running text: a space between two, except before closing punctuation, combining
    marks and a closing quote, after an opening bracket or quote, and around a slash."""
    if not tokens:
        return ""
    parts = [tokens[0]]
    quote_open = tokens[0] == '"'
    for previous, token in zip(tokens, tokens[1:]):
        closes_quote = token == '"' and quote_open
        if token == '"':
            quote_open = not quote_open
        opened_quote = previous == '"' and quote_open and not closes_quote
        attached = (token[0] in CLOSING or closes_quote or unicodedata.combining(token[0]) > 0
                    or previous[-1] in OPENING or opened_quote or token in JOINING or previous in JOINING)
        parts.append(token if attached else " " + token)
    return "".join(parts)
