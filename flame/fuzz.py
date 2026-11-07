"""The two string measures FLAME takes from rapidfuzz, in pure Python, for where rapidfuzz cannot be
installed, such as in Pyodide. They give the same values as rapidfuzz.

Both use bit-parallel algorithms over Python integers, one integer operation per character of the
second string, so long strings stay fast:
- `levenshtein_distance`: Myers (1999) in Hyyrö's formulation, as `rapidfuzz.distance.Levenshtein.distance`;
- `ratio`: the normalized Indel similarity times 100, from the longest common subsequence after
  Hyyrö (2004), as `rapidfuzz.fuzz.ratio` without a processor.
"""


def _match_masks(pattern: str) -> dict[str, int]:
    masks: dict[str, int] = {}
    for i, ch in enumerate(pattern):
        masks[ch] = masks.get(ch, 0) | (1 << i)
    return masks


def levenshtein_distance(a: str, b: str) -> int:
    """The number of insertions, deletions and substitutions that turn a into b."""
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0 or not b:
        return m
    masks = _match_masks(a)
    full = (1 << m) - 1
    last = 1 << (m - 1)
    positive, negative, score = full, 0, m
    for ch in b:
        eq = masks.get(ch, 0)
        xv = eq | negative
        xh = (((eq & positive) + positive) ^ positive) | eq
        ph = negative | ~(xh | positive)
        mh = positive & xh
        if ph & last:
            score += 1
        elif mh & last:
            score -= 1
        ph = (ph << 1) | 1
        mh <<= 1
        positive = (mh | ~(xv | ph)) & full
        negative = ph & xv & full
    return score


def lcs_length(a: str, b: str) -> int:
    """The length of the longest common subsequence of a and b."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    masks = _match_masks(a)
    full = (1 << len(a)) - 1
    v = full
    for ch in b:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return len(a) - v.bit_count()


class Levenshtein:
    """The name `rapidfuzz.distance.Levenshtein` gives this measure."""

    distance = staticmethod(levenshtein_distance)


def ratio(a: str, b: str) -> float:
    """100 * (1 - indel distance / (len(a) + len(b))); 100 for two empty strings."""
    total = len(a) + len(b)
    if total == 0:
        return 100.0
    return 100.0 * 2 * lcs_length(a, b) / total
