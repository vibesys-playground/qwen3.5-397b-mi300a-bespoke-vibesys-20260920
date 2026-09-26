"""Incremental prompt tokenization for multi-turn chat.

Every turn of a session resends the whole conversation, so the rendered chat
template of turn k+1 starts with the rendered template of turn k. Tokenizing
only the new tail is exact as long as the split point is the end of a special
token: HF tokenizers treat added and special tokens as atomic, so no merge
spans one. The cache therefore stores, per prompt it has seen, the longest
prefix of that prompt ending at a special token, together with its ids, in a
`capacity`-bounded LRU table.

`selfcheck` re-derives sample prompts incrementally and compares them with a
full tokenization. A tokenizer that fails it disables the cache rather than
risk prompt ids that differ from what the accuracy gate assumes.
"""

from __future__ import annotations

from collections import OrderedDict


class PromptCache:
    """LRU of (prompt prefix text -> token ids) used to skip re-tokenizing history."""

    def __init__(self, tok, capacity: int = 64) -> None:  # noqa: ANN001
        self.tok = tok
        self.capacity = capacity
        self.entries: OrderedDict[str, list[int]] = OrderedDict()
        self.marks = tuple({t for t in getattr(tok, "all_special_tokens", ()) if t})
        self.enabled = bool(self.marks)

    def encode(self, text: str) -> list[int]:
        """Token ids of `text`, tokenizing only the part no cached prefix covers."""
        if not self.enabled:
            return self.tokenize(text)
        base, ids = self._lookup(text)
        full = ids + self.tokenize(text[len(base) :])
        cut = self._safe_cut(text)
        if cut > len(base):
            self._store(text[:cut], ids + self.tokenize(text[len(base) : cut]))
        return full

    def tokenize(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"] if text else []

    def selfcheck(self, texts: list[str]) -> bool:
        """Disable the cache unless incremental encoding reproduces full encoding on `texts`.

        An empty `texts` fails: the caller could not produce probes, so nothing was checked.
        """
        probe = PromptCache(self.tok, self.capacity)
        self.enabled = (
            bool(self.marks)
            and bool(texts)
            and all(probe.enabled and probe.encode(t) == self.tokenize(t) for t in texts)
        )
        return self.enabled

    def _lookup(self, text: str) -> tuple[str, list[int]]:
        best = ""
        for key in self.entries:
            if len(best) < len(key) <= len(text) and text.startswith(key):
                best = key
        if not best:
            return "", []
        self.entries.move_to_end(best)
        return best, self.entries[best]

    def _safe_cut(self, text: str) -> int:
        """End offset of the last special token in `text`; 0 when it has none."""
        cut = 0
        for mark in self.marks:
            at = text.rfind(mark)
            cut = max(cut, at + len(mark)) if at >= 0 else cut
        return cut

    def _store(self, key: str, ids: list[int]) -> None:
        self.entries[key] = ids
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
