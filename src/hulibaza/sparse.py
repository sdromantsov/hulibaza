"""Deterministic sparse keyword vectors.

A term-frequency sparse vector: each distinct word maps to a stable index via
BLAKE2b, and its value is the raw term count. Qdrant applies IDF weighting at
query time (the collection's Modifier.IDF), so we store plain term frequencies.

Indices are deterministic across processes and Python versions — the indices
written at ingestion time must match those computed at query time after a
server restart, otherwise keyword search silently misses.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

# Sparse index space: blake2b(digest_size=4) yields 32 bits, folded into 2**31.
# The stored vectors depend on this exact value — changing it invalidates every
# collection's keyword index. The format is tagged "blake2b-idf" in state.
SPARSE_INDEX_SPACE = 2**31

_WORD_RE = re.compile(r"\b\w+\b")


@dataclass
class SparseVector:
    """Plain sparse vector (parallel indices + term-frequency values).

    Decoupled from qdrant_client so this stays unit-testable without infra; the
    Qdrant layer maps it to qdrant_client.models.SparseVector at upsert/query.
    """

    indices: list[int]
    values: list[float]


def token_to_index(token: str) -> int:
    """Stable hash of a token to a sparse index.

    BLAKE2b (cryptographic, fast, identical across processes and Python
    versions) guarantees ingestion-time and query-time indices agree.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % SPARSE_INDEX_SPACE


def build_sparse_vector(text: str) -> SparseVector:
    """Build a term-frequency sparse vector from text.

    Tokens are lowercased ``\\b\\w+\\b`` runs. Terms are sorted for a
    deterministic index/value order; the value is the raw occurrence count
    (IDF is applied server-side by Qdrant).
    """
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(tokens)
    sorted_terms = sorted(counts)
    indices = [token_to_index(t) for t in sorted_terms]
    values = [float(counts[t]) for t in sorted_terms]
    return SparseVector(indices=indices, values=values)
