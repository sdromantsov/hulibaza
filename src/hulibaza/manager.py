"""Serving orchestration: retrieval, navigation, ingest task registry, status.

Holds the shared clients (config, embedder, Qdrant, Postgres, ingestor) and
applies the two consistency gates (gates.py) on every search. Ingest runs as a
background asyncio task tracked in an in-memory run registry (the run-level
status pending|running|completed|failed lives here, not in Postgres).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from hulibaza.config import GlobalConfig, ResolvedSectionConfig, discover_sections
from hulibaza.embedding_client import EmbeddingError
from hulibaza.files import discover_files
from hulibaza.gates import check_completeness, check_validity
from hulibaza.health import ModelHealthRegistry
from hulibaza.ingest import Ingestor
from hulibaza.sparse import build_sparse_vector

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str], int]
_DENSE_MODES = ("hybrid", "semantic")
_VALID_MODES = ("hybrid", "semantic", "keyword")


@dataclass
class RunInfo:
    section: str
    status: str = "pending"  # pending | running | completed | failed
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    summary: dict | None = None


class HulibazaManager:
    def __init__(
        self,
        config: GlobalConfig,
        embedder,
        qdrant,
        state,
        ingestor: Ingestor,
        get_token_counter: Callable[[str], TokenCounter],
        *,
        now: Callable[[], float] = time.time,
        health: ModelHealthRegistry | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.qdrant = qdrant
        self.state = state
        self.ingestor = ingestor
        self.get_token_counter = get_token_counter
        self._now = now
        self.health = health or ModelHealthRegistry(embedder, now=now)
        self._runs: dict[str, RunInfo] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._orphan_task: asyncio.Task | None = None
        self._reconciled: set[str] = set()

    async def _cleanup_orphans(self) -> list[str]:
        """Sections tracked in PG whose directory is gone: drop the
        collection + PG rows. Gated on the ingest('all') request."""
        discovered = set(self._sections())
        orphans = sorted(set(await self.state.list_sections()) - discovered)
        for name in orphans:
            logger.info("Cleaning up orphan section '%s'", name)
            await self.qdrant.drop_collection(name)
            await self.state.delete_section(name)
        return orphans

    async def _reconcile_once(self, section_name: str) -> None:
        """Lazy reconcile on first touch: PG is authority, so re-assert
        every tombstoned row's in_use=false into Qdrant, healing any crash drift
        (PG tombstoned but the Qdrant mark never landed). Runs once per process."""
        if section_name in self._reconciled:
            return
        self._reconciled.add(section_name)
        for r in await self.state.list_files(section_name):
            if not r.in_use:
                await self.qdrant.set_in_use(section_name, r.rel_path, False)

    # ── section discovery ──

    def _sections(self) -> dict[str, ResolvedSectionConfig]:
        return {s.name: s for s in discover_sections(self.config)}

    def _disk_files(self, section: ResolvedSectionConfig) -> set[str]:
        result = discover_files(
            section.path,
            size_cap_text=self.config.size_cap_text_bytes,
            size_cap_other=self.config.size_cap_other_bytes,
        )
        return {c.rel_path for c in result.files}

    def sections(self) -> dict:
        """Compact: name, description, enabled + disabled_reason."""
        return {
            "sections": [
                {
                    "name": s.name,
                    "description": s.description,
                    "enabled": s.enabled,
                    "disabled_reason": s.disabled_reason,
                }
                for s in self._sections().values()
            ]
        }

    async def section_details(self, section_name: str) -> dict:
        """Heavy per-section, incl. ingestion coverage (recovery step)."""
        section = self._sections().get(section_name)
        if section is None:
            return {"error": f"Section '{section_name}' not found"}
        fp = await self.state.get_section(section_name)
        rows = await self.state.list_files(section_name)
        disk = self._disk_files(section) if section.enabled else set()
        comp = check_completeness(disk, rows)
        ingested = [r for r in rows if r.status == "ingested" and r.in_use]
        return {
            "name": section.name,
            "description": section.description,
            "enabled": section.enabled,
            "disabled_reason": section.disabled_reason,
            "embed_model": section.embed_model,
            "chunk_size": section.chunk_size,
            "chunk_overlap": section.chunk_overlap,
            "ingested_at": fp.ingested_at.isoformat() if fp and fp.ingested_at else None,
            "total_chunks": sum(r.total_chunks for r in ingested),
            "coverage": {
                "on_disk": len(disk),
                "ingested": len(ingested),
                "pending": len(comp.pending),
                "changed": len(comp.changed),
                "in_progress": len(comp.in_progress),
                "marked_for_deletion": sum(
                    1 for r in rows if not r.in_use and r.delete_after is not None
                ),
            },
        }

    # ── retrieval ──

    async def search(
        self,
        section_name: str,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid",
        allow_incomplete: bool = False,
    ) -> dict:
        if mode not in _VALID_MODES:
            return {"error": f"Unknown mode '{mode}'. Use one of {_VALID_MODES}."}
        section = self._sections().get(section_name)
        if section is None:
            return {"error": f"Section '{section_name}' not found"}
        if not section.enabled:
            return {"error": f"Section '{section_name}' is disabled: {section.disabled_reason}"}
        if not await self.qdrant.collection_exists(section.name):
            return {"error": f"Section '{section_name}' has not been ingested yet. Run ingest()."}

        await self._reconcile_once(section.name)  # lazy self-heal on first touch
        fingerprint = await self.state.get_section(section.name)
        warnings: list[str] = []

        # Validity gate — hard-blocks dense modes, never overridable.
        validity = check_validity(section, fingerprint)
        if not validity.valid:
            if mode in _DENSE_MODES:
                return {
                    "error": f"Validity gate: stored vectors are from different params "
                    f"({validity.reason}). '{mode}' is blocked; use mode='keyword' "
                    f"(model-independent) or re-ingest.",
                }
            warnings.append(f"validity mismatch ({validity.reason}); keyword-only")

        # Completeness gate — blocks all modes by default; allow_incomplete lifts it.
        rows = await self.state.list_files(section.name)
        completeness = check_completeness(self._disk_files(section), rows)
        if not completeness.complete:
            if not allow_incomplete:
                return {
                    "error": f"Section '{section_name}' is not fully ingested "
                    f"({completeness.summary()}). Re-run ingest(), or retry with "
                    f"allow_incomplete=true to search the in_use subset.",
                    "excluded": completeness.excluded(),
                }
            warnings.append(f"incomplete index ({completeness.summary()}); searching in_use subset")

        # Execute. Keyword needs no embedder; dense modes embed the query.
        sparse = build_sparse_vector(query)
        if mode == "keyword":
            results = await self.qdrant.keyword_search(section.name, sparse, limit=top_k)
        else:
            if not self.health.is_usable(section.embed_model):
                h = self.health.get(section.embed_model)
                return {
                    "error": f"Model '{section.embed_model}' is {h.status}"
                    + (f": {h.error}" if h.error else "")
                    + f". '{mode}' is unavailable; use mode='keyword' to query the "
                    f"existing index without the embedder.",
                }
            try:
                dense = await self.embedder.embed_single(section.embed_model, query)
            except EmbeddingError as e:
                self.health.mark_unhealthy(section.embed_model, str(e))
                return {
                    "error": f"Embedding failed for '{section.embed_model}': {e}. "
                    f"Model marked unhealthy; retry with mode='keyword' to query "
                    f"without the embedder.",
                }
            self.health.mark_healthy(section.embed_model, len(dense))
            if mode == "semantic":
                results = await self.qdrant.semantic_search(section.name, dense, limit=top_k)
            else:
                results = await self.qdrant.hybrid_search(section.name, dense, sparse, limit=top_k)

        response = {
            "section": section_name,
            "query": query,
            "mode": mode,
            "results": [
                {
                    "text": r.text,
                    "source_file": r.source_file,
                    "page_number": r.page_number,
                    "chunk_index": r.chunk_index,
                    "score": r.score,
                }
                for r in results
            ],
        }
        if warnings:
            response["warnings"] = warnings
        return response

    async def list_files(self, section_name: str) -> dict:
        section, err = await self._queryable(section_name)
        if err:
            return err
        files = await self.qdrant.list_files(section.name)
        return {
            "section": section_name,
            "files": [
                {"source_file": f.source_file, "total_chunks": f.total_chunks, "max_page": f.max_page}
                for f in files
            ],
        }

    async def get_chunks(self, section_name: str, source_file: str, start: int = 0, count: int = 10) -> dict:
        section, err = await self._queryable(section_name)
        if err:
            return err
        chunks = await self.qdrant.get_chunks_by_index(section.name, source_file, start, count)
        return {
            "section": section_name,
            "source_file": source_file,
            "chunks": [
                {"text": c.text, "page_number": c.page_number, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        }

    async def _queryable(self, section_name: str):
        section = self._sections().get(section_name)
        if section is None:
            return None, {"error": f"Section '{section_name}' not found"}
        if not section.enabled:
            return None, {"error": f"Section '{section_name}' is disabled: {section.disabled_reason}"}
        if not await self.qdrant.collection_exists(section.name):
            return None, {"error": f"Section '{section_name}' has not been ingested yet"}
        return section, None

    # ── ingest (background) + status ──

    def ingest(self, section: str = "all") -> dict:
        """Spawn background ingest task(s); return immediately."""
        sections = self._sections()
        if section == "all":
            targets = [s for s in sections.values() if s.enabled]
            self._orphan_task = asyncio.create_task(self._cleanup_orphans())
        else:
            cfg = sections.get(section)
            if cfg is None:
                return {"error": f"Section '{section}' not found"}
            if not cfg.enabled:
                return {"error": f"Section '{section}' is disabled: {cfg.disabled_reason}"}
            targets = [cfg]

        started, already = [], []
        for cfg in targets:
            task = self._tasks.get(cfg.name)
            if task is not None and not task.done():
                already.append(cfg.name)
                continue
            self._runs[cfg.name] = RunInfo(cfg.name, status="pending")
            self._tasks[cfg.name] = asyncio.create_task(self._run_ingest(cfg))
            started.append(cfg.name)
        return {"status": "started" if started else "already_running",
                "started": started, "already_running": already}

    async def _run_ingest(self, section: ResolvedSectionConfig) -> None:
        run = self._runs[section.name]
        run.status = "running"
        run.started_at = self._now()
        try:
            summary = await self.ingestor.ingest_section(
                section,
                size_cap_text=self.config.size_cap_text_bytes,
                size_cap_other=self.config.size_cap_other_bytes,
                token_counter=self.get_token_counter(section.embed_model),
            )
            run.summary = vars(summary)
            run.status = "failed" if summary.status == "failed" else "completed"
            run.error = summary.reason
        except Exception as e:  # pragma: no cover - defensive
            logger.exception("Ingest task crashed for '%s'", section.name)
            run.status = "failed"
            run.error = str(e)
        finally:
            run.finished_at = self._now()

    async def status(self, filters: dict | None = None) -> dict:
        """Run status (incl. failed + reason), pending counts, skips."""
        want = (lambda k: True) if not filters else (lambda k: k in filters)
        out: dict = {}
        sections = self._sections()

        if want("sections"):
            names = filters.get("sections") if filters else None
            sec_out = []
            for name, cfg in sections.items():
                if names and name not in names:
                    continue
                rows = await self.state.list_files(name)
                run = self._runs.get(name)
                sec_out.append({
                    "name": name,
                    "enabled": cfg.enabled,
                    "disabled_reason": cfg.disabled_reason,
                    "run_status": run.status if run else "pending",
                    "run_error": run.error if run else None,
                    "ingested": sum(1 for r in rows if r.status == "ingested" and r.in_use),
                    "errors": [
                        {"file": r.rel_path, "reason": r.error_reason}
                        for r in rows if r.status == "error"
                    ],
                    "skipped": [
                        {"file": r.rel_path, "status": r.status}
                        for r in rows if r.status in ("skipped_size", "skipped_binary")
                    ],
                    "marked_for_deletion": sum(
                        1 for r in rows if not r.in_use and r.delete_after is not None
                    ),
                })
            out["sections"] = sec_out

        if want("models"):
            names = filters.get("models") if filters else None
            out["models"] = self.health.snapshot(names)

        if want("health"):
            out["health"] = {
                "embedder": await self.embedder.health_check(),
                "qdrant": await self.qdrant.health_check(),
                "postgres": await self.state.health_check(),
            }
        return out
