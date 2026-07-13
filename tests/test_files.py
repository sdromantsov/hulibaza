"""Unit tests for file discovery & classification."""

from pathlib import Path

import pytest

from hulibaza.files import (
    classify_extension,
    discover_files,
    load_allow_exts,
    looks_binary,
)

pytestmark = pytest.mark.unit

CAPS = {"size_cap_text": 10_000, "size_cap_other": 1_000}


def _write(root: Path, rel: str, content: bytes | str = "hello") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        p.write_text(content, encoding="utf-8")
    else:
        p.write_bytes(content)
    return p


def _rels(result) -> set[str]:
    return {f.rel_path for f in result.files}


def _class_of(result, rel: str) -> str:
    return next(f.file_class for f in result.files if f.rel_path == rel)


# ── classification ──


def test_classify_known_binary_text_other(tmp_path):
    allow = load_allow_exts(tmp_path)
    assert classify_extension(".pdf", allow) == "known_binary"
    assert classify_extension(".PDF", allow) == "known_binary"
    assert classify_extension(".md", allow) == "known_text"
    assert classify_extension(".py", allow) == "known_text"
    assert classify_extension(".ts", allow) == "other"  # ambiguous, excluded
    assert classify_extension(".xyz", allow) == "other"
    assert classify_extension("", allow) == "other"


def test_known_text_class_assigned(tmp_path):
    _write(tmp_path, "a.md")
    _write(tmp_path, "b.unknownext")
    _write(tmp_path, "c.pdf", b"%PDF-1.4 fake")
    r = discover_files(tmp_path, **CAPS)
    assert _class_of(r, "a.md") == "known_text"
    assert _class_of(r, "b.unknownext") == "other"
    assert _class_of(r, "c.pdf") == "known_binary"


# ── ignore semantics ──


def test_hidden_dirs_and_junk_dirs_pruned(tmp_path):
    _write(tmp_path, "keep.md")
    _write(tmp_path, ".git/config", "x")
    _write(tmp_path, "node_modules/pkg/index.js", "x")
    _write(tmp_path, "sub/.venv/lib.py", "x")
    r = discover_files(tmp_path, **CAPS)
    assert _rels(r) == {"keep.md"}


def test_control_files_not_ingested(tmp_path):
    _write(tmp_path, "doc.md")
    _write(tmp_path, ".hulibazaignore", "*.log\n")
    _write(tmp_path, ".env", "SECRET=1")
    r = discover_files(tmp_path, **CAPS)
    assert _rels(r) == {"doc.md"}


def test_section_ignore_extends_default(tmp_path):
    _write(tmp_path, "keep.md")
    _write(tmp_path, "drop.md")
    _write(tmp_path, ".hulibazaignore", "drop.md\n")
    r = discover_files(tmp_path, **CAPS)
    assert _rels(r) == {"keep.md"}


def test_ignore_wins_over_allow(tmp_path):
    # .log is a known-text extension; an ignore glob must still exclude it.
    _write(tmp_path, "app.log", "line")
    _write(tmp_path, "keep.txt", "x")
    _write(tmp_path, ".hulibazaignore", "*.log\n")
    # .log isn't in the default allow anyway; add it via section allow to prove
    # ignore precedence even when explicitly allowed.
    _write(tmp_path, ".hulibazaallow", ".log\n")
    r = discover_files(tmp_path, **CAPS)
    assert _rels(r) == {"keep.txt"}


def test_negation_reincludes(tmp_path):
    _write(tmp_path, ".github/workflows/ci.yml", "on: push")
    _write(tmp_path, "readme.md")
    _write(tmp_path, ".hulibazaignore", "!.github/\n")
    r = discover_files(tmp_path, **CAPS)
    assert "readme.md" in _rels(r)
    assert ".github/workflows/ci.yml" in _rels(r)


# ── binary sniff + size caps ──


def test_binary_file_skipped(tmp_path):
    _write(tmp_path, "img.customext", b"GIF89a\x00\x00binary")
    _write(tmp_path, "ok.txt", "plain text")
    r = discover_files(tmp_path, **CAPS)
    assert _rels(r) == {"ok.txt"}
    assert any(s.rel_path == "img.customext" and s.reason == "binary" for s in r.skipped)


def test_size_cap_other_vs_text(tmp_path):
    # "other" cap is small (1000); a 2000-byte unknown-ext file is too large.
    _write(tmp_path, "big.unknownext", "x" * 2000)
    # known-text cap is larger (10000); a 2000-byte .md fits.
    _write(tmp_path, "big.md", "x" * 2000)
    r = discover_files(tmp_path, **CAPS)
    assert "big.md" in _rels(r)
    assert "big.unknownext" not in _rels(r)
    assert any(s.rel_path == "big.unknownext" and s.reason.startswith("too_large") for s in r.skipped)


def test_pdf_uncapped_and_unsniffed(tmp_path):
    # A PDF is binary (has null bytes) and larger than the "other" cap, yet
    # must still be discovered — known-binary is exempt from sniff and cap.
    _write(tmp_path, "big.pdf", b"%PDF-1.7\x00" + b"A" * 5000)
    r = discover_files(tmp_path, **CAPS)
    assert "big.pdf" in _rels(r)
    assert _class_of(r, "big.pdf") == "known_binary"


def test_section_allow_extends_default(tmp_path):
    _write(tmp_path, "weird.zzz", "text content")
    _write(tmp_path, ".hulibazaallow", ".zzz\n")
    r = discover_files(tmp_path, **CAPS)
    assert _class_of(r, "weird.zzz") == "known_text"


def test_discovery_is_sorted_deterministic(tmp_path):
    for name in ["c.md", "a.md", "b.md", "sub/z.md", "sub/a.md"]:
        _write(tmp_path, name)
    r = discover_files(tmp_path, **CAPS)
    rels = [f.rel_path for f in r.files]
    assert rels == sorted(rels)


def test_looks_binary(tmp_path):
    txt = _write(tmp_path, "a.txt", "no null bytes here")
    binf = _write(tmp_path, "b.bin", b"\x00\x01\x02")
    assert looks_binary(txt) is False
    assert looks_binary(binf) is True
