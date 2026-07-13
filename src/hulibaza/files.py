"""File discovery & classification.

Decides WHICH files under a section are ingested and into which class:

  - known_binary : has a dedicated parser (`.pdf`). Uncapped, never sniffed.
  - known_text   : extension listed in `.hulibazaallow`. Parsed UTF-8, larger cap.
  - other        : unknown / no extension. Parsed UTF-8 only if it passes the
                   null-byte sniff and the small cap.

Path filtering uses gitignore semantics: a shipped default ignore list plus an
optional per-section `.hulibazaignore` that EXTENDS it (`!` re-includes; ignore
wins over `.hulibazaallow` on overlap). Text *extraction* lives in `parser.py`.

Evaluation order per file: path ignore -> extension classify -> (non-binary:
size cap via stat, then null-byte sniff). The cap is checked before the sniff so
content is only read after a file passes the cheap stat gate.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from pathspec import GitIgnoreSpec

from hulibaza.parser import KNOWN_BINARY_EXTS

logger = logging.getLogger(__name__)

# Per-section override filenames.
IGNORE_FILENAME = ".hulibazaignore"
ALLOW_FILENAME = ".hulibazaallow"

# Bytes peeked to detect binary content via a null byte.
_SNIFF_BYTES = 8192

_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class FileCandidate:
    """A file that should be ingested, and its class."""

    path: Path
    rel_path: str  # POSIX path relative to the section root
    file_class: str  # "known_binary" | "known_text" | "other"


@dataclass(frozen=True)
class SkipRecord:
    """A file excluded for a surfaced reason (binary / too_large / unreadable).

    Files removed by `.hulibazaignore` are NOT recorded here — ignoring is
    intentional and silent. Skips are the junk-protection signal for status().
    """

    rel_path: str
    reason: str


@dataclass
class DiscoveryResult:
    files: list[FileCandidate] = field(default_factory=list)
    skipped: list[SkipRecord] = field(default_factory=list)


# ── shipped defaults, parsed once at import ──


def _read_data(name: str) -> str:
    return (_DATA_DIR / name).read_text(encoding="utf-8")


def _pattern_lines(text: str) -> list[str]:
    """Non-blank, non-comment lines (gitignore patterns), stripped."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _ext_lines(text: str) -> list[str]:
    """Extensions from a list file, normalized to lowercase with a leading dot."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("."):
            line = "." + line
        out.append(line.lower())
    return out


_DEFAULT_IGNORE_LINES: list[str] = _pattern_lines(_read_data("default_ignore"))
_DEFAULT_ALLOW_EXTS: frozenset[str] = frozenset(_ext_lines(_read_data("default_allow")))


def build_ignore_spec(section_dir: Path) -> GitIgnoreSpec:
    """Shipped default ignore patterns + the section's `.hulibazaignore`."""
    lines = list(_DEFAULT_IGNORE_LINES)
    section_ignore = section_dir / IGNORE_FILENAME
    if section_ignore.is_file():
        try:
            lines += _pattern_lines(section_ignore.read_text(encoding="utf-8"))
        except OSError as e:
            logger.warning("Cannot read %s: %s", section_ignore, e)
    return GitIgnoreSpec.from_lines(lines)


def load_allow_exts(section_dir: Path) -> frozenset[str]:
    """Shipped default text extensions + the section's `.hulibazaallow`."""
    exts = set(_DEFAULT_ALLOW_EXTS)
    section_allow = section_dir / ALLOW_FILENAME
    if section_allow.is_file():
        try:
            exts.update(_ext_lines(section_allow.read_text(encoding="utf-8")))
        except OSError as e:
            logger.warning("Cannot read %s: %s", section_allow, e)
    return frozenset(exts)


def classify_extension(ext: str, allow_exts: frozenset[str]) -> str:
    ext = ext.lower()
    if ext in KNOWN_BINARY_EXTS:
        return "known_binary"
    if ext in allow_exts:
        return "known_text"
    return "other"


def looks_binary(path: Path, peek: int = _SNIFF_BYTES) -> bool:
    """True if the first `peek` bytes contain a null byte.

    A null byte is a reliable binary signal: legitimate UTF-8 text never
    contains one. Unreadable files are treated as binary so discovery skips
    them gracefully instead of crashing.
    """
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(peek)
    except OSError as e:
        logger.warning("Cannot peek %s: %s — treating as binary", path, e)
        return True


def discover_files(
    section_dir: os.PathLike | str,
    *,
    size_cap_text: int,
    size_cap_other: int,
    allow_exts: frozenset[str] | None = None,
    ignore_spec: GitIgnoreSpec | None = None,
) -> DiscoveryResult:
    """Walk `section_dir` and classify every appropriate file.

    Discovery order is deterministic (sorted dirs and files) so the resulting
    file order is stable across runs. `size_cap_text` / `size_cap_other` are
    byte caps; known-binary files (PDF) are exempt.
    """
    section_dir = Path(section_dir)
    if allow_exts is None:
        allow_exts = load_allow_exts(section_dir)
    if ignore_spec is None:
        ignore_spec = build_ignore_spec(section_dir)

    result = DiscoveryResult()
    for root, dirnames, filenames in os.walk(section_dir):
        root_path = Path(root)

        # Prune ignored directories in place (respects `!` negations), sorted
        # for deterministic traversal.
        kept: list[str] = []
        for d in sorted(dirnames):
            rel = (root_path / d).relative_to(section_dir).as_posix() + "/"
            if not ignore_spec.match_file(rel):
                kept.append(d)
        dirnames[:] = kept

        for fname in sorted(filenames):
            path = root_path / fname
            rel = path.relative_to(section_dir).as_posix()
            if ignore_spec.match_file(rel):
                continue  # ignored files are silent, not skips

            file_class = classify_extension(path.suffix, allow_exts)
            if file_class == "known_binary":
                result.files.append(FileCandidate(path, rel, file_class))
                continue

            # known_text / other: cheap stat (cap) before reading (sniff).
            try:
                size = path.stat().st_size
            except OSError as e:
                result.skipped.append(SkipRecord(rel, f"unreadable:{e}"))
                continue
            cap = size_cap_text if file_class == "known_text" else size_cap_other
            if size > cap:
                result.skipped.append(SkipRecord(rel, f"too_large:{size}>{cap}"))
                continue
            if looks_binary(path):
                result.skipped.append(SkipRecord(rel, "binary"))
                continue
            result.files.append(FileCandidate(path, rel, file_class))

    return result
