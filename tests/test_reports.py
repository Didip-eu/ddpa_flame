"""The comparison report: text from a charter stays text in the HTML."""

import flame
import flame_tokens


def test_markup_in_a_text_is_escaped():
    text = 'Datum <b>Wienn</b> & anno domini millesimo trecentesimo, in die sancti Georii'
    tokens = flame_tokens.tokenize(text)
    left, right = flame.SimilarityVisualizer.highlight_similarities(tokens, tokens, 'p0', max_gap_words=5, fuzz_threshold=0.7)
    for side in (left, right):
        assert '<b>' not in side and '&lt;' in side and '&amp;' in side


def test_tokens_are_substrings_in_order():
    text = 'Ich Jacob Ebser vergich (mit dem brief), das mir 4/7 "pfunt" phenning.'
    tokens = flame_tokens.tokenize(text)
    assert flame._token_char_spans(tokens, text) is not None
    assert flame_tokens.detokenize(tokens) == 'Ich Jacob Ebser vergich (mit dem brief), das mir 4/7 "pfunt" phenning.'
