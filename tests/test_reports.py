"""The comparison report: text from a charter stays text in the HTML."""

from nltk.tokenize import word_tokenize

import flame


def test_markup_in_a_text_is_escaped():
    text = 'Datum <b>Wienn</b> & anno domini millesimo trecentesimo, in die sancti Georii'
    tokens = word_tokenize(text)
    left, right = flame.SimilarityVisualizer.highlight_similarities(tokens, tokens, 'p0', max_gap_words=5, fuzz_threshold=0.7)
    for side in (left, right):
        assert '<b>' not in side and '&lt;' in side and '&amp;' in side
