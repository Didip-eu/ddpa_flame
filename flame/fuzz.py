"""The string measures FLAME takes from rapidfuzz, in pure Python, for where rapidfuzz cannot be
installed, such as in Pyodide. They give the same values as rapidfuzz.

They use bit-parallel algorithms over Python integers, one integer operation per character of the
second string, so long strings stay fast:
- `levenshtein_distance`: Myers (1999) in Hyyrö's formulation, as `rapidfuzz.distance.Levenshtein.distance`;
- `ratio`: the normalized Indel similarity times 100, from the longest common subsequence after
  Hyyrö (2004), as `rapidfuzz.fuzz.ratio` without a processor;
- `partial_ratio`: the best of that similarity between the shorter string and the parts of the
  longer one that rapidfuzz scores, as `rapidfuzz.fuzz.partial_ratio` without a processor.
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


def _masked_lcs_length(masks: dict[str, int], length: int, text: str) -> int:
    full = (1 << length) - 1
    v = full
    for ch in text:
        u = v & masks.get(ch, 0)
        v = ((v + u) | (v - u)) & full
    return length - v.bit_count()


def lcs_length(a: str, b: str) -> int:
    """The length of the longest common subsequence of a and b."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return 0
    return _masked_lcs_length(_match_masks(a), len(a), b)


class Levenshtein:
    """The name `rapidfuzz.distance.Levenshtein` gives this measure."""

    distance = staticmethod(levenshtein_distance)


def ratio(a: str, b: str) -> float:
    """100 * (1 - indel distance / (len(a) + len(b))); 100 for two empty strings."""
    total = len(a) + len(b)
    if total == 0:
        return 100.0
    return 100.0 * 2 * lcs_length(a, b) / total


def _best_part_similarity(needle: str, text: str) -> float:
    """The best normalized Indel similarity of needle against a part of text.

    The parts are rapidfuzz's: the prefixes shorter than needle, every window of its length, and
    the suffixes from the last window on.
    """
    length = len(needle)
    masks = _match_masks(needle)
    last_start = len(text) - length
    parts = [(0, end) for end in range(1, length)]
    parts += [(start, start + length) for start in range(last_start)]
    parts += [(start, len(text)) for start in range(last_start, len(text))]
    best = 0.0
    for start, end in parts:
        maximum = length + end - start
        distance = maximum - 2 * _masked_lcs_length(masks, length, text[start:end])
        # rapidfuzz's arithmetic, so the score is the same float to the last bit
        best = max(best, 1.0 - distance / maximum)
    return best


def partial_ratio(a: str, b: str) -> float:
    """The best `ratio` of the shorter string against a part of the longer one; 100 for two empty strings."""
    if not a and not b:
        return 100.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if not shorter:
        return 0.0
    best = _best_part_similarity(shorter, longer)
    if best != 1.0 and len(a) == len(b):
        best = max(best, _best_part_similarity(longer, shorter))
    return best * 100
