"""Exact local token counting via HuggingFace tokenizers.

One tokenizer.json per declared model, loaded once at startup. `count(text)`
runs in microseconds (Rust under the hood), so the chunker calls it directly per
block with no caching. Counts include the special tokens (BOS/EOS/...) the
post-processor adds — the same budget the embedding server sees at inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class LocalTokenizer:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._tokenizer = Tokenizer.from_file(self._path)
        logger.info("Loaded tokenizer from %s", self._path)

    def count(self, text: str) -> int:
        """Number of tokens (including special tokens) in `text`."""
        return len(self._tokenizer.encode(text).ids)

    def __call__(self, text: str) -> int:
        return self.count(text)


class TokenizerRegistry:
    """One LocalTokenizer per declared model, keyed by model name."""

    def __init__(self, specs: dict[str, str]) -> None:
        """specs: {model_name: tokenizer_path}."""
        self._tokenizers = {name: LocalTokenizer(path) for name, path in specs.items()}

    def get(self, model: str) -> LocalTokenizer:
        if model not in self._tokenizers:
            raise KeyError(f"no tokenizer loaded for model '{model}'")
        return self._tokenizers[model]

    def __contains__(self, model: str) -> bool:
        return model in self._tokenizers
