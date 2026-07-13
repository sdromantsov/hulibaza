"""Unit tests for deterministic sparse keyword vectors."""

import hashlib

import pytest

from hulibaza.sparse import (
    SPARSE_INDEX_SPACE,
    SparseVector,
    build_sparse_vector,
    token_to_index,
)

pytestmark = pytest.mark.unit


def _reference_index(token: str) -> int:
    """Independent reimplementation of the documented hash, to catch drift."""
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(d, "big") % SPARSE_INDEX_SPACE


# ── token_to_index ──


def test_index_matches_documented_algorithm():
    for tok in ["hello", "world", "cudamalloc", "функция", "a"]:
        assert token_to_index(tok) == _reference_index(tok)


def test_index_locked_values():
    # Frozen values: if these change, the stored keyword index is invalidated.
    assert token_to_index("hello") == 1785136887
    assert token_to_index("world") == 13193140


def test_index_in_range():
    for tok in ["x", "supercalifragilistic", "0xdeadbeef", "тест"]:
        assert 0 <= token_to_index(tok) < SPARSE_INDEX_SPACE


def test_index_deterministic():
    assert token_to_index("repeat") == token_to_index("repeat")


# ── build_sparse_vector ──


def test_empty_and_whitespace():
    assert build_sparse_vector("") == SparseVector([], [])
    assert build_sparse_vector("   \n\t ") == SparseVector([], [])
    assert build_sparse_vector("!!! ... ---") == SparseVector([], [])


def test_term_frequency_counts():
    vec = build_sparse_vector("alpha beta alpha alpha beta gamma")
    # term -> count: alpha 3, beta 2, gamma 1
    by_index = dict(zip(vec.indices, vec.values))
    assert by_index[token_to_index("alpha")] == 3.0
    assert by_index[token_to_index("beta")] == 2.0
    assert by_index[token_to_index("gamma")] == 1.0


def test_lowercased():
    vec = build_sparse_vector("The THE the")
    assert vec.indices == [token_to_index("the")]
    assert vec.values == [3.0]


def test_word_boundaries_split_punctuation():
    vec = build_sparse_vector("foo, bar. foo-bar (foo)")
    by_index = dict(zip(vec.indices, vec.values))
    # foo x3 (foo, foo, foo), bar x2 (bar, bar from foo-bar)
    assert by_index[token_to_index("foo")] == 3.0
    assert by_index[token_to_index("bar")] == 2.0


def test_indices_ordered_by_sorted_terms():
    text = "gamma alpha beta"
    vec = build_sparse_vector(text)
    expected = [token_to_index(t) for t in sorted(["gamma", "alpha", "beta"])]
    assert vec.indices == expected


def test_deterministic_across_calls():
    text = "Deterministic sparse vector must be identical across calls."
    a = build_sparse_vector(text)
    b = build_sparse_vector(text)
    assert a == b


def test_parallel_indices_values_length():
    vec = build_sparse_vector("one two three two one one")
    assert len(vec.indices) == len(vec.values) == 3  # 3 distinct terms
