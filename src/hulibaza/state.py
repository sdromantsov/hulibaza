"""Postgres state: section fingerprints + per-file tracking.

Two tables (see the ERD):

  section — one row per section: the embedding fingerprint (embed_model,
    chunk_size, chunk_overlap) plus schema-format guards (sizing_mode,
    sparse_format) and ingested_at. The run-level status is NOT stored here; it
    is a server/task-registry concern.

  file — one row per tracked file: content_hash, size, mtime (fast-path),
    total_chunks/chunks_done (per-batch resume checkpoint), in_use +
    delete_after (soft-delete tombstone), and status
    (ingesting | ingested | error | skipped_size | skipped_binary).

PG is the authority. `status='ingesting'` is a write-ahead intent: a row stuck
there after a crash is the partial to clean and resume. Section mutual exclusion
is a Postgres advisory lock (cross-process — the purge runs as a separate
process), held on a dedicated connection for the life of the critical section.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

import psycopg

logger = logging.getLogger(__name__)

SIZING_MODE = "tokens"
SPARSE_FORMAT = "blake2b-idf"

FILE_STATUSES = ("ingesting", "ingested", "error", "skipped_size", "skipped_binary")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS section (
    section_name  TEXT PRIMARY KEY,
    embed_model   TEXT NOT NULL,
    chunk_size    INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    sizing_mode   TEXT NOT NULL DEFAULT 'tokens',
    sparse_format TEXT NOT NULL DEFAULT 'blake2b-idf',
    ingested_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS file (
    file_id       BIGSERIAL PRIMARY KEY,
    section_name  TEXT NOT NULL REFERENCES section(section_name) ON DELETE CASCADE,
    rel_path      TEXT NOT NULL,
    content_hash  TEXT NOT NULL DEFAULT '',
    size          BIGINT NOT NULL DEFAULT 0,
    mtime         TIMESTAMPTZ,
    total_chunks  INTEGER NOT NULL DEFAULT 0,
    chunks_done   INTEGER NOT NULL DEFAULT 0,
    in_use        BOOLEAN NOT NULL DEFAULT TRUE,
    status        TEXT NOT NULL,
    error_reason  TEXT,
    delete_after  TIMESTAMPTZ,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (section_name, rel_path)
);

CREATE INDEX IF NOT EXISTS file_section_hash_idx ON file (section_name, content_hash);
"""


@dataclass
class SectionFingerprint:
    section_name: str
    embed_model: str
    chunk_size: int
    chunk_overlap: int
    sizing_mode: str
    sparse_format: str
    ingested_at: datetime | None


@dataclass
class FileRow:
    section_name: str
    rel_path: str
    content_hash: str
    size: int
    mtime: datetime | None
    total_chunks: int
    chunks_done: int
    in_use: bool
    status: str
    error_reason: str | None
    delete_after: datetime | None
    updated_at: datetime


_FILE_COLS = (
    "section_name, rel_path, content_hash, size, mtime, total_chunks, "
    "chunks_done, in_use, status, error_reason, delete_after, updated_at"
)


def _row_to_file(row) -> FileRow:
    return FileRow(*row)


def _lock_key(section_name: str) -> int:
    """Stable signed 64-bit key for pg_advisory_lock from a section name."""
    digest = hashlib.blake2b(section_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class StateStore:
    def __init__(self, postgres_url: str) -> None:
        self.postgres_url = postgres_url

    async def _connect(self) -> psycopg.AsyncConnection:
        # autocommit: each statement is its own transaction. The WAL-style
        # ordering (Qdrant upsert THEN chunks_done bump) is enforced by the
        # ingestion orchestration, not by multi-statement PG transactions.
        return await psycopg.AsyncConnection.connect(self.postgres_url, autocommit=True)

    async def init_schema(self) -> None:
        async with await self._connect() as conn:
            await conn.execute(_SCHEMA_SQL)
        logger.info("Postgres schema ensured")

    # ── section fingerprint ──

    async def ensure_section(
        self, section_name: str, embed_model: str, chunk_size: int, chunk_overlap: int
    ) -> None:
        """Upsert the section fingerprint. Preserves ingested_at."""
        async with await self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO section
                    (section_name, embed_model, chunk_size, chunk_overlap,
                     sizing_mode, sparse_format)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (section_name) DO UPDATE SET
                    embed_model   = EXCLUDED.embed_model,
                    chunk_size    = EXCLUDED.chunk_size,
                    chunk_overlap = EXCLUDED.chunk_overlap,
                    sizing_mode   = EXCLUDED.sizing_mode,
                    sparse_format = EXCLUDED.sparse_format
                """,
                (section_name, embed_model, chunk_size, chunk_overlap, SIZING_MODE, SPARSE_FORMAT),
            )

    async def mark_section_ingested(self, section_name: str) -> None:
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE section SET ingested_at = NOW() WHERE section_name = %s",
                (section_name,),
            )

    async def get_section(self, section_name: str) -> SectionFingerprint | None:
        async with await self._connect() as conn:
            cur = await conn.execute(
                "SELECT section_name, embed_model, chunk_size, chunk_overlap, "
                "sizing_mode, sparse_format, ingested_at FROM section WHERE section_name = %s",
                (section_name,),
            )
            row = await cur.fetchone()
        return SectionFingerprint(*row) if row else None

    async def delete_section(self, section_name: str) -> None:
        """Drop the section and (via cascade) all its file rows."""
        async with await self._connect() as conn:
            await conn.execute("DELETE FROM section WHERE section_name = %s", (section_name,))

    async def list_sections(self) -> list[str]:
        async with await self._connect() as conn:
            cur = await conn.execute("SELECT section_name FROM section ORDER BY section_name")
            return [r[0] for r in await cur.fetchall()]

    # ── file rows ──

    async def get_file(self, section_name: str, rel_path: str) -> FileRow | None:
        async with await self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_FILE_COLS} FROM file WHERE section_name = %s AND rel_path = %s",
                (section_name, rel_path),
            )
            row = await cur.fetchone()
        return _row_to_file(row) if row else None

    async def list_files(self, section_name: str) -> list[FileRow]:
        async with await self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_FILE_COLS} FROM file WHERE section_name = %s ORDER BY rel_path",
                (section_name,),
            )
            return [_row_to_file(r) for r in await cur.fetchall()]

    async def begin_file_ingest(
        self,
        section_name: str,
        rel_path: str,
        content_hash: str,
        size: int,
        mtime: float,
        total_chunks: int,
    ) -> None:
        """Write the WAL intent: mark the file 'ingesting', reset chunks_done,
        record the new fingerprint (content_hash/size/mtime/total_chunks)."""
        async with await self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO file
                    (section_name, rel_path, content_hash, size, mtime, total_chunks,
                     chunks_done, in_use, status, error_reason, delete_after, updated_at)
                VALUES (%s, %s, %s, %s, to_timestamp(%s), %s, 0, TRUE, 'ingesting', NULL, NULL, NOW())
                ON CONFLICT (section_name, rel_path) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    size         = EXCLUDED.size,
                    mtime        = EXCLUDED.mtime,
                    total_chunks = EXCLUDED.total_chunks,
                    chunks_done  = 0,
                    in_use       = TRUE,
                    status       = 'ingesting',
                    error_reason = NULL,
                    delete_after = NULL,
                    updated_at   = NOW()
                """,
                (section_name, rel_path, content_hash, size, mtime, total_chunks),
            )

    async def set_chunks_done(self, section_name: str, rel_path: str, chunks_done: int) -> None:
        """Advance the per-batch resume checkpoint."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET chunks_done = %s, updated_at = NOW() "
                "WHERE section_name = %s AND rel_path = %s",
                (chunks_done, section_name, rel_path),
            )

    async def complete_file_ingest(self, section_name: str, rel_path: str) -> None:
        """Commit: chunks_done = total_chunks, status 'ingested', in_use true."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET status = 'ingested', chunks_done = total_chunks, "
                "in_use = TRUE, delete_after = NULL, error_reason = NULL, updated_at = NOW() "
                "WHERE section_name = %s AND rel_path = %s",
                (section_name, rel_path),
            )

    async def mark_file_error(self, section_name: str, rel_path: str, reason: str) -> None:
        """Record a per-file failure (fail-soft). Upserts so a failure before any
        row exists (e.g. parse error) is still surfaced in status()."""
        async with await self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO file
                    (section_name, rel_path, content_hash, size, mtime, total_chunks,
                     chunks_done, in_use, status, error_reason, delete_after, updated_at)
                VALUES (%s, %s, '', 0, NULL, 0, 0, FALSE, 'error', %s, NULL, NOW())
                ON CONFLICT (section_name, rel_path) DO UPDATE SET
                    status = 'error', error_reason = EXCLUDED.error_reason, updated_at = NOW()
                """,
                (section_name, rel_path, reason),
            )

    async def record_skip(
        self, section_name: str, rel_path: str, status: str, reason: str, size: int = 0
    ) -> None:
        """Record a skipped file (skipped_size | skipped_binary) with its reason."""
        if status not in ("skipped_size", "skipped_binary"):
            raise ValueError(f"not a skip status: {status}")
        async with await self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO file
                    (section_name, rel_path, content_hash, size, mtime, total_chunks,
                     chunks_done, in_use, status, error_reason, delete_after, updated_at)
                VALUES (%s, %s, '', %s, NULL, 0, 0, FALSE, %s, %s, NULL, NOW())
                ON CONFLICT (section_name, rel_path) DO UPDATE SET
                    size = EXCLUDED.size, in_use = FALSE, status = EXCLUDED.status,
                    error_reason = EXCLUDED.error_reason, updated_at = NOW()
                """,
                (section_name, rel_path, size, status, reason),
            )

    # ── tombstone / move ──

    async def tombstone_file(self, section_name: str, rel_path: str, grace_days: int) -> None:
        """Soft-delete: in_use=false, delete_after = now + grace."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET in_use = FALSE, "
                "delete_after = NOW() + make_interval(days => %s), updated_at = NOW() "
                "WHERE section_name = %s AND rel_path = %s",
                (grace_days, section_name, rel_path),
            )

    async def mark_changed(self, section_name: str, rel_path: str) -> None:
        """Changed file: hide the old vectors (in_use=false) with NO grace —
        ingest rebuilds them and purge must never reap them (delete_after NULL)."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET in_use = FALSE, delete_after = NULL, updated_at = NOW() "
                "WHERE section_name = %s AND rel_path = %s",
                (section_name, rel_path),
            )

    async def restore_file(self, section_name: str, rel_path: str) -> None:
        """Un-tombstone: in_use=true, clear delete_after (e.g. file reappeared)."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET in_use = TRUE, delete_after = NULL, updated_at = NOW() "
                "WHERE section_name = %s AND rel_path = %s",
                (section_name, rel_path),
            )

    async def find_by_hash(self, section_name: str, content_hash: str) -> list[FileRow]:
        """Files in the section with a given content_hash (move detection)."""
        async with await self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_FILE_COLS} FROM file "
                f"WHERE section_name = %s AND content_hash = %s ORDER BY rel_path",
                (section_name, content_hash),
            )
            return [_row_to_file(r) for r in await cur.fetchall()]

    async def repath_file(self, section_name: str, old_rel: str, new_rel: str) -> None:
        """Move: rename a file row's rel_path, restoring it to in_use."""
        async with await self._connect() as conn:
            await conn.execute(
                "UPDATE file SET rel_path = %s, in_use = TRUE, delete_after = NULL, "
                "updated_at = NOW() WHERE section_name = %s AND rel_path = %s",
                (new_rel, section_name, old_rel),
            )

    # ── reconcile / purge support ──

    async def list_ingesting(self, section_name: str) -> list[FileRow]:
        """Files stuck at status='ingesting' — crash orphans to clean/resume."""
        async with await self._connect() as conn:
            cur = await conn.execute(
                f"SELECT {_FILE_COLS} FROM file "
                f"WHERE section_name = %s AND status = 'ingesting' ORDER BY rel_path",
                (section_name,),
            )
            return [_row_to_file(r) for r in await cur.fetchall()]

    async def list_expired_tombstones(self, section_name: str | None = None) -> list[FileRow]:
        """Tombstones whose grace has elapsed (in_use=false, delete_after<=now).
        Consumed by the separate purge process."""
        sql = (
            f"SELECT {_FILE_COLS} FROM file "
            "WHERE in_use = FALSE AND delete_after IS NOT NULL AND delete_after <= NOW()"
        )
        params: tuple = ()
        if section_name is not None:
            sql += " AND section_name = %s"
            params = (section_name,)
        sql += " ORDER BY section_name, rel_path"
        async with await self._connect() as conn:
            cur = await conn.execute(sql, params)
            return [_row_to_file(r) for r in await cur.fetchall()]

    async def delete_file_row(self, section_name: str, rel_path: str) -> None:
        """Hard-delete a file row (purge, after Qdrant points are gone)."""
        async with await self._connect() as conn:
            await conn.execute(
                "DELETE FROM file WHERE section_name = %s AND rel_path = %s",
                (section_name, rel_path),
            )

    # ── advisory lock (cross-process section mutual exclusion) ──

    @asynccontextmanager
    async def try_section_lock(self, section_name: str):
        """Yield True if the section lock was acquired (held on a dedicated
        connection for the block), False if another holder has it."""
        key = _lock_key(section_name)
        conn = await psycopg.AsyncConnection.connect(self.postgres_url, autocommit=True)
        acquired = False
        try:
            cur = await conn.execute("SELECT pg_try_advisory_lock(%s)", (key,))
            acquired = (await cur.fetchone())[0]
            if acquired:
                try:
                    yield True
                finally:
                    await conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
            else:
                yield False
        finally:
            await conn.close()

    async def health_check(self) -> bool:
        try:
            async with await self._connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception:
            return False
