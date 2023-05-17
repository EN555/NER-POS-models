from collections import Counter, OrderedDict
from typing import List, Iterable


def build_vocab_from_iterator(iterator: Iterable, max_tokens: int, specials: List[str]):
    vocab_counter = Counter(iterator)
    vocab_counter = sorted(vocab_counter.items(), key=lambda x: (-x[1], x[0]))
    if max_tokens is None:
        vocab = OrderedDict(vocab_counter)
    else:
        vocab = OrderedDict(vocab_counter[: max_tokens - len(specials)])
    return Vocab(list(vocab.values()) + specials)


class Vocab:
    def __init__(self, words: List[str]):
        self.default_index = None
        self.words = words
        self.word_to_index = {word: idx for idx, word in enumerate(words)}

    def set_default_index(self, idx: int):
        self.default_index = idx

    def lookup_token(self, idx: int):
        if idx < 0 or idx >= len(self.words):
            raise ValueError(f"index out of range {idx} in lookup_token")
        return self.words[idx]

    def __getitem__(self, word: str):
        res_idx = self.word_to_index.get(word)
        if res_idx is None:
            if self.default_index:
                return self.default_index
            else:
                raise ValueError(f"word out of range word in __getitem__")
        return res_idx
