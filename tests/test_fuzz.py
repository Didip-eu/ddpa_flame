"""flame.fuzz against rapidfuzz, which it stands in for where rapidfuzz cannot be installed."""

import random

import pytest

from flame import fuzz

native = pytest.importorskip("rapidfuzz.fuzz")

CHOSEN_PAIRS = [
    pytest.param("", "", id="both empty"),
    pytest.param("", "abc", id="first empty"),
    pytest.param("abc", "", id="second empty"),
    pytest.param("a", "b", id="nothing shared"),
    pytest.param("this is a test", "this is a test!", id="shorter inside the longer"),
    pytest.param("damus et concedimus ac presentibus", "et concedimus", id="shorter second"),
    pytest.param("concedimus ac", "damus et concedimus", id="shorter running past the end"),
    pytest.param("ac presentibus", "presentibus et", id="shorter running past the start"),
    pytest.param("abcd", "dcba", id="equal lengths"),
    pytest.param("sigillo", "sigillo", id="equal strings"),
    pytest.param("vnd ſancte æcclesie", "und sancte ecclesie", id="non-ASCII letters"),
    pytest.param("é a", "é a", id="combining accent"),
    pytest.param("\U0001d51e\U0001d51f", "a\U0001d51e\U0001d51fb", id="beyond the BMP"),
    pytest.param("ab" * 40, "b" * 30 + "ab" * 39 + "c" * 30, id="needle longer than 64"),
]


@pytest.mark.parametrize(("a", "b"), CHOSEN_PAIRS)
def test_partial_ratio_equals_rapidfuzz_on_chosen_pairs(a, b):
    assert fuzz.partial_ratio(a, b) == native.partial_ratio(a, b)


def test_partial_ratio_equals_rapidfuzz_on_random_pairs():
    rng = random.Random(0)
    for _ in range(2000):
        alphabet = rng.choice(["ab", "abcdefgh", "etaoin srhldcu", "aeiouäöüßſ \U0001d51e"])
        lengths = [rng.choice([rng.randint(0, 8), rng.randint(0, 40), rng.randint(50, 120)]) for _ in range(2)]
        if rng.random() < 0.2:
            lengths[1] = lengths[0]
        a, b = ("".join(rng.choice(alphabet) for _ in range(length)) for length in lengths)
        assert fuzz.partial_ratio(a, b) == native.partial_ratio(a, b), (a, b)
