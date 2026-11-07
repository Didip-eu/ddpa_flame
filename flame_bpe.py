"""A byte-pair-encoding tokenizer in pure Python, trained the way Hugging Face `tokenizers` trains
`BPE` with a `Whitespace` pre-tokenizer and a `BpeTrainer` (no normalizer, no prefixes or suffixes,
no alphabet limit, minimum frequency 0), so that it yields the same vocabulary, merges and tokens.

It stands in for `tokenizers` where that Rust package cannot be installed, such as in Pyodide.

Training follows `tokenizers/src/models/bpe/trainer.rs`:
- the alphabet is every character of the corpus, given ids by code point after the special tokens;
- pairs are counted over the distinct words, weighted by word frequency;
- a max-heap pops the most frequent pair, ties going to the smaller pair of ids; an entry whose
  count is stale is re-pushed with the current count;
- after a merge, the pairs that gained occurrences are pushed again;
- training stops when the vocabulary reaches `vocab_size` or no pair occurs;
- a pair merged a second time (it can return through another entry) takes the later rank, as the
  merges list of a saved model does when it is loaded again.

Encoding applies the merges by rank, lowest rank first and leftmost first.
"""

import heapq
from collections import Counter, defaultdict

import regex

# `\w` as the Rust regex crate defines it, which includes combining marks; `re`'s `\w` does not.
WORD = r"[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]"
WHITESPACE = regex.compile(rf"{WORD}+|(?:(?!{WORD})[^\s])+")


def pre_tokenize(text: str) -> list[str]:
    """The words of `Whitespace`: runs of word characters, and runs of other non-space characters."""
    return WHITESPACE.findall(text)


def _merge_word(symbols: list[int], first: int, second: int, replacement: int) -> list[tuple[tuple[int, int], int]]:
    """Merge every occurrence of (first, second) in place, left to right, and return the pair count
    changes, as `Word::merge` of `tokenizers` does."""
    changes = []
    i = 0
    while i < len(symbols):
        if symbols[i] == first and i + 1 < len(symbols) and symbols[i + 1] == second:
            if i > 0:
                changes.append(((symbols[i - 1], first), -1))
                changes.append(((symbols[i - 1], replacement), 1))
            symbols[i:i + 2] = [replacement]
            if i < len(symbols) - 1:
                changes.append(((second, symbols[i + 1]), -1))
                changes.append(((replacement, symbols[i + 1]), 1))
        i += 1
    return changes


class BpeTokenizer:
    """A trained vocabulary and its merges; `encode` returns the token strings of a text."""

    def __init__(self, unk_token: str = "[UNK]"):
        self.unk_token = unk_token
        self.vocab: dict[str, int] = {}
        self.id_to_token: list[str] = []
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        self._cache: dict[str, list[str]] = {}

    def _add(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        return self.vocab[token]

    def train(self, lines, vocab_size: int, special_tokens=()) -> "BpeTokenizer":
        word_counts = Counter()
        for line in lines:
            word_counts.update(pre_tokenize(line))

        for token in special_tokens:
            self._add(token)
        alphabet = Counter()
        for word, count in word_counts.items():
            for ch in word:
                alphabet[ch] += count
        for ch in sorted(alphabet, key=ord):
            self._add(ch)

        words = [[self.vocab[ch] for ch in word] for word in word_counts]
        counts = list(word_counts.values())

        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for index, (symbols, count) in enumerate(zip(words, counts)):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += count
                where[pair].add(index)

        order = 0
        queue = []
        for pair, count in pair_counts.items():
            if count > 0:
                queue.append((-count, pair, order, where[pair]))
                order += 1
        heapq.heapify(queue)

        rank = 0
        while queue and len(self.vocab) < vocab_size:
            negative, pair, entry, positions = heapq.heappop(queue)
            if -negative != pair_counts[pair]:
                heapq.heappush(queue, (-pair_counts[pair], pair, entry, positions))
                continue
            if -negative < 1:
                break

            new_id = self._add(self.id_to_token[pair[0]] + self.id_to_token[pair[1]])
            self.merges[pair] = (rank, new_id)
            rank += 1

            gained = defaultdict(set)
            for index in positions:
                for changed, delta in _merge_word(words[index], pair[0], pair[1], new_id):
                    pair_counts[changed] += delta * counts[index]
                    if delta > 0:
                        gained[changed].add(index)
            for changed, indexes in gained.items():
                if pair_counts[changed] > 0:
                    heapq.heappush(queue, (-pair_counts[changed], changed, order, indexes))
                    order += 1
        return self

    def _encode_word(self, word: str) -> list[str]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        unk = self.vocab.get(self.unk_token)
        symbols = []
        for ch in word:
            token_id = self.vocab.get(ch, unk)
            if token_id is not None:
                symbols.append(token_id)
        while len(symbols) > 1:
            best = None
            for i, pair in enumerate(zip(symbols, symbols[1:])):
                merge = self.merges.get(pair)
                if merge is not None and (best is None or merge[0] < best[0]):
                    best = (merge[0], i, merge[1])
            if best is None:
                break
            _, i, new_id = best
            symbols[i:i + 2] = [new_id]
        tokens = [self.id_to_token[token_id] for token_id in symbols]
        self._cache[word] = tokens
        return tokens

    def encode(self, text: str) -> list[str]:
        return [token for word in pre_tokenize(text) for token in self._encode_word(word)]
