"""The display tokens of the reports."""

import flame
import flame_tokens


def test_tokens_are_substrings_in_order():
    text = 'Ich Jacob Ebser vergich (mit dem brief), das mir 4/7 "pfunt" phenning.'
    tokens = flame_tokens.tokenize(text)
    assert flame._token_char_spans(tokens, text) is not None
    assert flame_tokens.detokenize(tokens) == 'Ich Jacob Ebser vergich (mit dem brief), das mir 4/7 "pfunt" phenning.'
