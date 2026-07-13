"""The two orthogonal consistency gates for retrieval.

Validity — an embedding-parameter mismatch between the section config and the
stored fingerprint means the stored vectors were produced by a *different*
model/params. Semantic & hybrid are hard-blocked (not overridable); keyword
still works (sparse vectors are model-independent).

Completeness — the index doesn't fully reflect disk: a pending (new / errored),
changed (daemon-tombstoned but still present), in-progress (mid-ingest), or
stale (indexed but gone from disk) file. Blocks all modes by default; lifted by
`allow_incomplete=true`, which returns the in_use subset plus a note of what's
excluded. `allow_incomplete` never lifts validity.

Both are pure functions of the section config, the fingerprint, the on-disk file
set (names only — no hashing), and the Postgres file rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hulibaza.config import ResolvedSectionConfig
from hulibaza.state import FileRow, SectionFingerprint

# File statuses that mean a disk file is intentionally accounted for.
_HANDLED_SKIP = ("skipped_size", "skipped_binary")


@dataclass
class ValidityVerdict:
    valid: bool
    reason: str | None = None


@dataclass
class CompletenessVerdict:
    complete: bool
    pending: list[str] = field(default_factory=list)  # new / errored — not indexed
    changed: list[str] = field(default_factory=list)  # tombstoned but still on disk
    in_progress: list[str] = field(default_factory=list)  # mid-ingest
    stale: list[str] = field(default_factory=list)  # indexed but gone from disk

    def excluded(self, sample: int = 10) -> dict:
        """Excluded files by category. Each list is capped to `sample` paths (a
        section can have hundreds pending mid-ingest) with a "...(+N more)"
        sentinel so this never floods a client's context. `summary()` still
        carries the exact counts."""
        def cap(items: list[str]) -> list[str]:
            if len(items) <= sample:
                return items
            return items[:sample] + [f"...(+{len(items) - sample} more)"]
        return {
            "pending": cap(self.pending),
            "changed": cap(self.changed),
            "in_progress": cap(self.in_progress),
            "stale": cap(self.stale),
        }

    def summary(self) -> str:
        parts = []
        for label, items in (
            ("pending", self.pending), ("changed", self.changed),
            ("in_progress", self.in_progress), ("stale", self.stale),
        ):
            if items:
                parts.append(f"{len(items)} {label}")
        return ", ".join(parts) if parts else "complete"


def check_validity(
    section: ResolvedSectionConfig, fingerprint: SectionFingerprint | None
) -> ValidityVerdict:
    """Do the section's embedding params still match the stored vectors?

    A None fingerprint (never ingested) is treated as valid here — the
    not-ingested case is handled by the caller before gating.
    """
    if fingerprint is None:
        return ValidityVerdict(True)
    mismatches = []
    if fingerprint.embed_model != section.embed_model:
        mismatches.append(f"embed_model {fingerprint.embed_model!r} -> {section.embed_model!r}")
    if fingerprint.chunk_size != section.chunk_size:
        mismatches.append(f"chunk_size {fingerprint.chunk_size} -> {section.chunk_size}")
    if fingerprint.chunk_overlap != section.chunk_overlap:
        mismatches.append(f"chunk_overlap {fingerprint.chunk_overlap} -> {section.chunk_overlap}")
    if mismatches:
        return ValidityVerdict(False, "; ".join(mismatches))
    return ValidityVerdict(True)


def check_completeness(disk_files: set[str], rows: list[FileRow]) -> CompletenessVerdict:
    """Compare on-disk file names against Postgres rows (no hashing).

    New files are known only from disk (the daemon does not create rows for
    them); changes are surfaced via the daemon's in_use=false mark on a still-
    present file; deletions show as an indexed row whose file is gone.
    """
    by_path = {r.rel_path: r for r in rows}
    v = CompletenessVerdict(complete=True)

    for r in rows:
        if r.status == "ingesting":
            v.in_progress.append(r.rel_path)

    for f in sorted(disk_files):
        row = by_path.get(f)
        if row is None:
            v.pending.append(f)  # new, not yet ingested
        elif row.status in _HANDLED_SKIP:
            continue  # intentional exclusion
        elif row.status == "ingesting":
            continue  # already in in_progress
        elif row.status == "ingested" and row.in_use:
            continue  # good
        elif row.status == "ingested" and not row.in_use:
            v.changed.append(f)  # daemon-tombstoned but still on disk
        else:  # error, or ingested-but-otherwise
            v.pending.append(f)

    for r in rows:
        if r.status == "ingested" and r.in_use and r.rel_path not in disk_files:
            v.stale.append(r.rel_path)  # indexed but gone from disk

    v.in_progress.sort()
    v.stale.sort()
    v.complete = not (v.pending or v.changed or v.in_progress or v.stale)
    return v
