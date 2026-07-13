"""Unit tests for content hashing & deterministic point IDs."""

import hashlib
import uuid

import pytest

from hulibaza.identity import (
    NAMESPACE,
    content_hash_bytes,
    content_hash_file,
    point_id,
)

pytestmark = pytest.mark.unit

BASE = ("docs", "guide/intro.md", "a" * 64, 0)


# ── namespace (locked) ──


def test_namespace_locked():
    # If this changes, every previously stored point ID is orphaned.
    assert str(NAMESPACE) == "435d3986-fa80-54b7-9476-79b9f05633c6"


# ── content hashing ──


def test_content_hash_bytes_matches_sha256():
    assert content_hash_bytes(b"hello") == hashlib.sha256(b"hello").hexdigest()


def test_content_hash_file_matches_bytes(tmp_path):
    data = b"some file contents\n" * 1000
    f = tmp_path / "doc.bin"
    f.write_bytes(data)
    assert content_hash_file(f) == content_hash_bytes(data)


def test_content_hash_file_empty(tmp_path):
    f = tmp_path / "empty"
    f.write_bytes(b"")
    assert content_hash_file(f) == hashlib.sha256(b"").hexdigest()


# ── point IDs ──


def test_point_id_deterministic():
    assert point_id(*BASE) == point_id(*BASE)


def test_point_id_is_valid_uuid5():
    pid = point_id(*BASE)
    u = uuid.UUID(pid)
    assert u.version == 5


def test_point_id_locked_sample():
    assert point_id(*BASE) == "490a9819-a5f9-53fc-8c44-ebb1b84b3d79"


def test_point_id_varies_by_chunk_index():
    assert point_id("docs", "a.md", "h" * 64, 0) != point_id("docs", "a.md", "h" * 64, 1)


def test_point_id_varies_by_content_hash():
    # Same file, changed content -> different IDs -> old and new coexist.
    assert point_id("docs", "a.md", "1" * 64, 0) != point_id("docs", "a.md", "2" * 64, 0)


def test_point_id_varies_by_source_file():
    assert point_id("docs", "a.md", "h" * 64, 0) != point_id("docs", "b.md", "h" * 64, 0)


def test_point_id_varies_by_section():
    # Identical-content file in two sections -> separated.
    assert point_id("s1", "a.md", "h" * 64, 0) != point_id("s2", "a.md", "h" * 64, 0)


def test_point_id_no_delimiter_collision():
    # NUL-join must not let field boundaries be ambiguous: ("a","b") vs ("ab","")
    # style shifts across the section/source_file boundary produce distinct IDs.
    a = point_id("sec", "file.md", "h" * 64, 0)
    b = point_id("secfile.md", "", "h" * 64, 0)
    assert a != b
