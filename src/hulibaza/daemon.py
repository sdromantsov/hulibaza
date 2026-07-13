"""Background lifecycle daemon.

An asyncio task that, every `daemon_poll_seconds`, tracks each section's files
by hash (with an mtime+size fast-path so unchanged files are never re-hashed).
It performs only cheap, reversible marks — it NEVER embeds or ingests:

  changed  -> eagerly tombstone the old vectors (in_use=false, no grace);
              ingest later deletes + rebuilds them.
  deleted  -> tombstone (in_use=false, delete_after = now + grace).
  moved    -> repath the points under new IDs, reuse vectors (no re-embed).
  new      -> not recorded (surfaced as a pending count by the gates).

Sections under the advisory lock (mid-ingest or mid-purge) are skipped. Hashing
of appeared files happens only when a deletion is detected (a possible move), so
genuinely-new files are never hashed here.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone

from hulibaza.config import GlobalConfig, ResolvedSectionConfig, discover_sections
from hulibaza.files import discover_files
from hulibaza.identity import content_hash_file
from hulibaza.ingest import repath_points

logger = logging.getLogger(__name__)

# mtime comparison tolerance (s). A false "changed" only forces a harmless
# re-hash that then finds the hash unchanged, so a loose bound is safe.
_MTIME_EPS = 1e-3


class Daemon:
    def __init__(self, config: GlobalConfig, state, qdrant) -> None:
        self.config = config
        self.state = state
        self.qdrant = qdrant
        self._stop = asyncio.Event()

    def _sections(self) -> list[ResolvedSectionConfig]:
        return [s for s in discover_sections(self.config) if s.enabled]

    async def run_forever(self) -> None:
        if not self.config.daemon_enabled:
            logger.info("Daemon disabled")
            return
        logger.info("Daemon started (poll=%ss)", self.config.daemon_poll_seconds)
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:  # pragma: no cover - defensive
                logger.exception("Daemon poll failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.config.daemon_poll_seconds)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> None:
        for section in self._sections():
            await self.poll_section(section)

    async def poll_section(self, section: ResolvedSectionConfig) -> None:
        """One mark-only pass over a section (skips it if advisory-locked)."""
        async with self.state.try_section_lock(section.name) as acquired:
            if not acquired:
                logger.debug("Daemon skips locked section '%s'", section.name)
                return
            await self._scan(section)

    async def _scan(self, section: ResolvedSectionConfig) -> None:
        discovery = discover_files(
            section.path,
            size_cap_text=self.config.size_cap_text_bytes,
            size_cap_other=self.config.size_cap_other_bytes,
        )
        disk = {c.rel_path: c for c in discovery.files}
        rows = await self.state.list_files(section.name)
        by_path = {r.rel_path: r for r in rows}

        # 1) Changes among currently-indexed, in_use files (fast-path first).
        for rel, cand in disk.items():
            row = by_path.get(rel)
            if row is None or not (row.status == "ingested" and row.in_use):
                continue
            if not self._looks_changed(cand, row):
                continue
            if content_hash_file(cand.path) != row.content_hash:
                logger.info("Daemon: '%s/%s' changed -> tombstone old (no grace)", section.name, rel)
                await self.state.mark_changed(section.name, rel)
                await self.qdrant.set_in_use(section.name, rel, False)

        # 2) Gone (indexed in_use file no longer on disk): a move or a deletion.
        gone = [r for r in rows if r.status == "ingested" and r.in_use and r.rel_path not in disk]
        if not gone:
            return

        appeared = [rel for rel in disk if rel not in by_path]  # new-or-moved-to paths
        appeared_hashes = {rel: content_hash_file(disk[rel].path) for rel in appeared}
        gone_by_hash: dict[str, list] = {}
        for r in gone:
            gone_by_hash.setdefault(r.content_hash, []).append(r)

        consumed = set()
        for rel, h in appeared_hashes.items():
            bucket = gone_by_hash.get(h)
            if bucket:
                src = bucket.pop(0)
                logger.info("Daemon: move '%s' -> '%s'", src.rel_path, rel)
                await repath_points(self.qdrant, self.state, section.name, src.rel_path, rel, h)
                consumed.add(src.rel_path)

        # 3) Remaining gone rows are real deletions -> tombstone with grace.
        for r in gone:
            if r.rel_path in consumed:
                continue
            logger.info("Daemon: '%s/%s' deleted -> tombstone (+%dd)",
                        section.name, r.rel_path, self.config.deletion_grace_days)
            await self.state.tombstone_file(section.name, r.rel_path, self.config.deletion_grace_days)
            await self.qdrant.set_in_use(section.name, r.rel_path, False)

    @staticmethod
    def _looks_changed(cand, row) -> bool:
        if row.size != cand.path.stat().st_size:
            return True
        if row.mtime is None:
            return True
        stored = row.mtime.replace(tzinfo=row.mtime.tzinfo or timezone.utc).timestamp()
        return abs(stored - cand.path.stat().st_mtime) > _MTIME_EPS
