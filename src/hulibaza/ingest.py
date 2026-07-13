"""Ingestion pipeline: the integrative core.

Wires discovery -> parse -> chunk -> embed -> sparse + deterministic ID ->
Qdrant upsert -> Postgres checkpoint, per section, under the section's advisory
lock. Crash-safety comes from a per-batch WAL:

    begin_file_ingest (status='ingesting', chunks_done=0)   [intent]
    for each batch:  Qdrant upsert  ->  bump chunks_done      [vectors before checkpoint]
    complete_file_ingest (status='ingested')                 [commit]

Invariant: a committed row implies its vectors exist. On restart, a same-hash
file resumes at chunks_done (<=1 batch redone, overwritten via the deterministic
ID); a changed file (new hash) restarts — its old vectors are deleted first
(delete + re-add), so nothing leaks.

Per-file dispatch:
    new        -> ingest fresh
    unchanged  -> skip (ingested, same hash, in_use)
    reappeared -> restore (ingested, same hash, tombstoned) — no re-embed
    resume     -> continue (ingesting, same hash) from chunks_done
    changed    -> delete old vectors, ingest fresh
    moved      -> repath: rewrite points under new IDs, no re-embed
Failures are per-file (fail-soft): status='error' + reason, run continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable

from hulibaza.chunker import Chunk, chunk_document, chunk_pages
from hulibaza.config import ResolvedSectionConfig
from hulibaza.embedding_client import EmbeddingClient, EmbeddingError
from hulibaza.files import discover_files
from hulibaza.identity import content_hash_file, point_id
from hulibaza.parser import parse_file
from hulibaza.qdrant_store import ChunkPoint, QdrantStore
from hulibaza.sparse import build_sparse_vector
from hulibaza.state import SectionFingerprint, StateStore

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]

_SKIP_REASON_TO_STATUS = {"too_large": "skipped_size", "binary": "skipped_binary"}


@dataclass
class IngestSummary:
    section: str
    status: str = "completed"  # completed | already_running | failed
    reason: str | None = None
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    resumed: int = 0
    moved: int = 0
    restored: int = 0
    errored: int = 0
    skipped: int = 0
    reset: bool = False


async def repath_points(qdrant, state, section_name: str, old_rel: str, new_rel: str, content_hash: str) -> None:
    """Move a file's points to new (path-derived) IDs reusing stored vectors —
    no re-embed — then delete the old points and repath the PG row. Shared by
    ingest and the daemon."""
    old_points = await qdrant.get_file_points(section_name, old_rel)
    new_points = [
        replace(p, id=point_id(section_name, new_rel, content_hash, p.chunk_index),
                source_file=new_rel, in_use=True)
        for p in old_points
    ]
    if new_points:
        await qdrant.upsert_chunks(section_name, new_points)
    await qdrant.delete_by_file(section_name, old_rel)
    await state.repath_file(section_name, old_rel, new_rel)


def decide_reset(stored: SectionFingerprint | None, section: ResolvedSectionConfig) -> str | None:
    """Reason to drop the collection and re-ingest from scratch, or None."""
    if stored is None:
        return None  # first ingest: create, don't "reset"
    if stored.embed_model != section.embed_model:
        return f"embed_model changed: '{stored.embed_model}' -> '{section.embed_model}'"
    if stored.chunk_size != section.chunk_size:
        return f"chunk_size changed: {stored.chunk_size} -> {section.chunk_size}"
    if stored.chunk_overlap != section.chunk_overlap:
        return f"chunk_overlap changed: {stored.chunk_overlap} -> {section.chunk_overlap}"
    return None


class Ingestor:
    def __init__(
        self,
        state: StateStore,
        qdrant: QdrantStore,
        embedder: EmbeddingClient,
        *,
        deletion_grace_days: int = 7,
    ) -> None:
        self.state = state
        self.qdrant = qdrant
        self.embedder = embedder
        self.deletion_grace_days = deletion_grace_days

    async def ingest_section(
        self,
        section: ResolvedSectionConfig,
        *,
        size_cap_text: int,
        size_cap_other: int,
        token_counter: TokenCounter,
    ) -> IngestSummary:
        """Ingest one section under its advisory lock. Returns a summary; the
        lock makes concurrent runs (or a purge) mutually exclusive."""
        async with self.state.try_section_lock(section.name) as acquired:
            if not acquired:
                return IngestSummary(section.name, status="already_running")
            return await self._ingest_locked(section, size_cap_text, size_cap_other, token_counter)

    async def _ingest_locked(
        self,
        section: ResolvedSectionConfig,
        size_cap_text: int,
        size_cap_other: int,
        token_counter: TokenCounter,
    ) -> IngestSummary:
        summary = IngestSummary(section.name)

        # Pre-flight: the embedder must answer (this also gives the vector dim).
        try:
            embed_dim = await self.embedder.get_embedding_dim(section.embed_model)
        except EmbeddingError as e:
            return IngestSummary(section.name, status="failed", reason=str(e))

        # Reset if an embedding parameter changed since the last fingerprint.
        stored = await self.state.get_section(section.name)
        reset_reason = decide_reset(stored, section)
        if reset_reason:
            logger.info("Reset '%s': %s", section.name, reset_reason)
            await self.qdrant.drop_collection(section.name)
            await self.state.delete_section(section.name)  # cascades file rows
            summary.reset = True
        await self.state.ensure_section(
            section.name, section.embed_model, section.chunk_size, section.chunk_overlap
        )
        await self.qdrant.ensure_collection(section.name, embed_dim)

        # Discover, and record skips (surfaced in status()).
        discovery = discover_files(
            section.path, size_cap_text=size_cap_text, size_cap_other=size_cap_other
        )
        for skip in discovery.skipped:
            kind = skip.reason.split(":", 1)[0]
            status = _SKIP_REASON_TO_STATUS.get(kind)
            if status:
                await self.state.record_skip(section.name, skip.rel_path, status, skip.reason)
            else:  # unreadable / other -> error
                await self.state.mark_file_error(section.name, skip.rel_path, skip.reason)
            summary.skipped += 1

        current_paths = {fc.rel_path for fc in discovery.files}
        for candidate in discovery.files:
            try:
                await self._process_one(section, candidate, current_paths, token_counter, summary)
            except Exception as e:  # fail-soft per file
                logger.exception("Ingest failed for %s/%s", section.name, candidate.rel_path)
                await self.state.mark_file_error(section.name, candidate.rel_path, str(e))
                summary.errored += 1

        await self.state.mark_section_ingested(section.name)
        return summary

    async def _process_one(self, section, candidate, current_paths, token_counter, summary) -> None:
        rel_path = candidate.rel_path
        stat = candidate.path.stat()
        content_hash = content_hash_file(candidate.path)
        existing = await self.state.get_file(section.name, rel_path)

        if existing is None:
            moved_from = await self._find_move_source(section.name, content_hash, rel_path, current_paths)
            if moved_from is not None:
                await self._repath(section.name, moved_from.rel_path, rel_path, content_hash)
                summary.moved += 1
                return
            await self._ingest_fresh(section, candidate, content_hash, stat, token_counter)
            summary.new += 1
            return

        same_hash = existing.content_hash == content_hash
        if existing.status == "ingested" and same_hash and existing.in_use:
            summary.unchanged += 1
            return
        if existing.status == "ingested" and same_hash and not existing.in_use:
            # Reappeared at the same path with the same content — just un-tombstone.
            await self.qdrant.set_in_use(section.name, rel_path, True)
            await self.state.restore_file(section.name, rel_path)
            summary.restored += 1
            return
        if existing.status == "ingesting" and same_hash:
            await self._ingest_resume(section, candidate, content_hash, stat, existing.chunks_done, token_counter)
            summary.resumed += 1
            return

        # Changed (new hash) or a prior error/partial to retry: delete any points
        # at this path, then ingest fresh. Parse BEFORE deleting so a broken new
        # version doesn't destroy the old vectors.
        chunks = self._parse_and_chunk(section, candidate, token_counter)
        await self.qdrant.delete_by_file(section.name, rel_path)
        await self._store(section, candidate, content_hash, stat, chunks, fresh=True, resume_from=0)
        summary.changed += 1

    async def _find_move_source(self, section_name, content_hash, new_rel, current_paths):
        """An ingested file with this hash whose old path is gone = a move."""
        for row in await self.state.find_by_hash(section_name, content_hash):
            if row.rel_path != new_rel and row.rel_path not in current_paths and row.status == "ingested":
                return row
        return None

    async def _repath(self, section_name, old_rel, new_rel, content_hash) -> None:
        await repath_points(self.qdrant, self.state, section_name, old_rel, new_rel, content_hash)

    def _parse_and_chunk(self, section, candidate, token_counter) -> list[Chunk]:
        doc = parse_file(candidate.path, section.path)
        if doc.pages and doc.pages[0].page_number > 0:
            return chunk_pages(
                pages=[(p.text, p.page_number) for p in doc.pages],
                source_file=doc.source_file,
                chunk_size=section.chunk_size,
                chunk_overlap=section.chunk_overlap,
                token_counter=token_counter,
            )
        return chunk_document(
            text=doc.full_text,
            source_file=doc.source_file,
            chunk_size=section.chunk_size,
            chunk_overlap=section.chunk_overlap,
            page_number=0,
            token_counter=token_counter,
        )

    async def _ingest_fresh(self, section, candidate, content_hash, stat, token_counter) -> None:
        chunks = self._parse_and_chunk(section, candidate, token_counter)
        await self._store(section, candidate, content_hash, stat, chunks, fresh=True, resume_from=0)

    async def _ingest_resume(self, section, candidate, content_hash, stat, chunks_done, token_counter) -> None:
        chunks = self._parse_and_chunk(section, candidate, token_counter)
        await self._store(section, candidate, content_hash, stat, chunks, fresh=False, resume_from=chunks_done)

    async def _store(
        self, section, candidate, content_hash, stat, chunks, *, fresh: bool, resume_from: int
    ) -> None:
        rel_path = candidate.rel_path
        total = len(chunks)
        if fresh:
            await self.state.begin_file_ingest(
                section.name, rel_path, content_hash, stat.st_size, stat.st_mtime, total
            )
            start = 0
        else:
            start = min(resume_from, total)

        batch_size = section.embed_batch_size
        i = start
        while i < total:
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            try:
                dense = await self.embedder.embed(section.embed_model, texts)
            except EmbeddingError:
                raise  # bubble to fail-soft handler; leaves chunks_done at last commit
            points = [
                ChunkPoint(
                    id=point_id(section.name, rel_path, content_hash, c.chunk_index),
                    text=c.text,
                    dense=vec,
                    sparse=build_sparse_vector(c.text),
                    source_file=rel_path,
                    page_number=c.page_number,
                    chunk_index=c.chunk_index,
                    section_name=section.name,
                    in_use=True,
                )
                for c, vec in zip(batch, dense)
            ]
            await self.qdrant.upsert_chunks(section.name, points)  # vectors first...
            await self.state.set_chunks_done(section.name, rel_path, i + len(batch))  # ...then checkpoint
            i += batch_size

        await self.state.complete_file_ingest(section.name, rel_path)
