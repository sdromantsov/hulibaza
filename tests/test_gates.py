"""Unit tests for the consistency gates — pure functions, no infra."""

from datetime import datetime, timezone

import pytest

from hulibaza.config import ResolvedSectionConfig
from hulibaza.gates import check_completeness, check_validity
from hulibaza.state import FileRow, SectionFingerprint

pytestmark = pytest.mark.unit

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _section(embed_model="m", chunk_size=512, chunk_overlap=50):
    return ResolvedSectionConfig(
        name="docs", path="/x", description="", embed_model=embed_model,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunk_overlap_ratio=0.1,
        headroom_ratio=0.02, embed_batch_size=16,
    )


def _fp(embed_model="m", chunk_size=512, chunk_overlap=50):
    return SectionFingerprint("docs", embed_model, chunk_size, chunk_overlap,
                              "tokens", "blake2b-idf", NOW)


def _row(rel, status="ingested", in_use=True, delete_after=None):
    return FileRow("docs", rel, "h" * 64, 1, None, 1, 1, in_use, status, None, delete_after, NOW)


# ── validity ──


def test_validity_none_fingerprint_is_valid():
    assert check_validity(_section(), None).valid


def test_validity_match():
    assert check_validity(_section(), _fp()).valid


def test_validity_embed_model_mismatch():
    v = check_validity(_section(embed_model="new"), _fp(embed_model="old"))
    assert not v.valid and "embed_model" in v.reason


def test_validity_chunk_size_mismatch():
    v = check_validity(_section(chunk_size=256), _fp(chunk_size=512))
    assert not v.valid and "chunk_size" in v.reason


def test_validity_chunk_overlap_mismatch():
    assert not check_validity(_section(chunk_overlap=10), _fp(chunk_overlap=50)).valid


# ── completeness ──


def test_complete_when_all_ingested():
    rows = [_row("a.md"), _row("b.md")]
    v = check_completeness({"a.md", "b.md"}, rows)
    assert v.complete and v.summary() == "complete"


def test_new_file_is_pending():
    v = check_completeness({"a.md", "new.md"}, [_row("a.md")])
    assert not v.complete and v.pending == ["new.md"]


def test_tombstoned_but_present_is_changed():
    v = check_completeness({"a.md"}, [_row("a.md", in_use=False)])
    assert not v.complete and v.changed == ["a.md"]


def test_ingesting_is_in_progress():
    v = check_completeness({"a.md"}, [_row("a.md", status="ingesting")])
    assert not v.complete and v.in_progress == ["a.md"]


def test_error_file_is_pending():
    v = check_completeness({"a.md"}, [_row("a.md", status="error", in_use=False)])
    assert not v.complete and v.pending == ["a.md"]


def test_skipped_files_are_handled_not_incomplete():
    rows = [_row("ok.md"), _row("big.log", status="skipped_size", in_use=False)]
    v = check_completeness({"ok.md", "big.log"}, rows)
    assert v.complete


def test_indexed_but_gone_is_stale():
    v = check_completeness(set(), [_row("gone.md")])
    assert not v.complete and v.stale == ["gone.md"]


def test_deleted_and_tombstoned_is_consistent():
    # in_use=false + delete_after set + not on disk = correctly excluded deletion.
    v = check_completeness(set(), [_row("del.md", in_use=False, delete_after=NOW)])
    assert v.complete
