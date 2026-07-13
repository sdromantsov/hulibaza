"""MCP server entry point. Exposes the retrieval tools via FastMCP.

Grounded retrieval: returns matching chunks; the client LLM composes the answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP

from hulibaza.config import load_global_config
from hulibaza.daemon import Daemon
from hulibaza.embedding_client import EmbeddingClient
from hulibaza.ingest import Ingestor
from hulibaza.local_tokenizer import TokenizerRegistry
from hulibaza.manager import HulibazaManager
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

SERVER_INSTRUCTIONS = """\
hulibaza — self-hosted grounded retrieval over "sections" (document collections;
one directory, one embedding model each). Returns matching chunks with
provenance; you compose the answer. Not Q&A. Every tool returns a JSON object;
failures carry an "error" field.

Workflow:
  1. sections() once — what exists and which are usable.
  2. search(section, query) — query is a topic/term/phrase, NOT a sentence.
  3. On a "not fully ingested" error: section_details(section) shows coverage;
     retry with allow_incomplete=true to search the available subset.
  4. Read structurally with list_files + get_chunks. Cite source_file/page; if
     nothing relevant, say so — don't invent.

Modes: hybrid (default, dense+sparse RRF) | semantic (dense) | keyword (sparse).
Search returns only current (in_use) chunks and applies two gates, both raised
as errors: validity (embedding params changed since indexing -> dense blocked,
keyword still works) and completeness (pending/changed/in-progress files -> all
modes blocked unless allow_incomplete=true). Keyword also works with the
embedder down.

Query style — terms not questions: "cudaMalloc", "PoolManager singleton"; not
"how does CUDA allocate memory".
"""

_manager: HulibazaManager | None = None


async def build_manager() -> HulibazaManager:
    config = load_global_config()
    embedder = EmbeddingClient(config.embedding_url, timeout=config.embedding_timeout)
    qdrant = QdrantStore(config.qdrant_url)
    state = StateStore(config.postgres_url)
    await state.init_schema()
    tokenizers = TokenizerRegistry({n: s.tokenizer_path for n, s in config.models.items()})
    ingestor = Ingestor(state, qdrant, embedder, deletion_grace_days=config.deletion_grace_days)
    return HulibazaManager(
        config, embedder, qdrant, state, ingestor,
        get_token_counter=lambda model: tokenizers.get(model).count,
    )


@contextlib.asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[None]:
    """Build the manager and start the background daemon at startup; stop it and
    close clients at shutdown. Ingestion is never auto-started."""
    global _manager
    _manager = await build_manager()
    logger.info("HulibazaManager initialized")

    # Check every declared model up front, then recheck unhealthy ones.
    await _manager.health.check_all(_manager.config.models)
    recheck_stop = asyncio.Event()
    recheck_task = asyncio.create_task(
        _manager.health.run_recheck_loop(
            _manager.config.defaults.model_retry_interval_seconds, recheck_stop
        )
    )

    daemon = Daemon(_manager.config, _manager.state, _manager.qdrant)
    daemon_task = asyncio.create_task(daemon.run_forever())
    try:
        yield
    finally:
        daemon.stop()
        recheck_stop.set()
        for task in (daemon_task, recheck_task):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        with contextlib.suppress(Exception):
            await _manager.embedder.aclose()


mcp = FastMCP(
    "hulibaza", instructions=SERVER_INSTRUCTIONS, host="0.0.0.0", port=8080, lifespan=_lifespan
)


async def _get_manager() -> HulibazaManager:
    global _manager
    if _manager is None:  # fallback if a tool runs outside the lifespan
        _manager = await build_manager()
        logger.info("HulibazaManager initialized (lazy)")
    return _manager


@mcp.tool()
async def sections() -> dict:
    """List sections. Call once. Returns {sections:[{name, description, enabled,
    disabled_reason}]}. A disabled section (invalid config: unknown embed_model,
    or chunk_size beyond the model's capacity) can't be searched or ingested."""
    return (await _get_manager()).sections()


@mcp.tool()
async def section_details(section: str) -> dict:
    """One section's config + ingestion coverage — the recovery step after a
    'not fully ingested' error. Returns {name, description, enabled,
    disabled_reason, embed_model, chunk_size, chunk_overlap, ingested_at,
    total_chunks, coverage:{on_disk, ingested, pending, changed, in_progress,
    marked_for_deletion}}."""
    return await (await _get_manager()).section_details(section)


@mcp.tool()
async def search(
    section: str,
    query: str,
    top_k: int = 3,
    mode: str = "hybrid",
    allow_incomplete: bool = False,
) -> dict:
    """Retrieve top-k chunks. `query` = topic/term/phrase, not a sentence. mode:
    hybrid | semantic | keyword (see server instructions for modes + the
    validity/completeness gates; allow_incomplete=true lifts completeness only).
    Returns {section, query, mode, results:[{text, source_file, page_number,
    chunk_index, score}], warnings?} or {error}."""
    return await (await _get_manager()).search(
        section, query, top_k=top_k, mode=mode, allow_incomplete=allow_incomplete
    )


@mcp.tool()
async def list_files(section: str) -> dict:
    """List a section's current (in_use) files. Returns {section,
    files:[{source_file, total_chunks, max_page}]} (max_page 0 = non-paginated,
    >=1 = PDF page count)."""
    return await (await _get_manager()).list_files(section)


@mcp.tool()
async def get_chunks(section: str, source_file: str, start: int = 0, count: int = 10) -> dict:
    """Read a file's chunks by index, ordered. Returns {section, source_file,
    chunks:[{text, page_number, chunk_index}]} for chunk_index in [start,
    start+count) (in_use only; fewer near EOF). Peek a head (start=0) or expand
    context around a search hit's chunk_index."""
    return await (await _get_manager()).get_chunks(section, source_file, start, count)


@mcp.tool()
async def ingest(section: str = "all") -> dict:
    """Index a section (or "all") so it becomes searchable. Background +
    incremental — only new/changed/moved files; an embedding-param change
    re-indexes; "all" also drops orphan sections (directory gone). Returns
    {status: started|already_running, started:[...], already_running:[...]} or
    {error}. Poll status()/section_details() for progress."""
    return (await _get_manager()).ingest(section)


@mcp.tool()
async def status(filters: dict | None = None) -> dict:
    """Operational state, scoped by `filters` (presence of a key includes that
    block; no filters = all). Keys -> blocks:
      "sections":[names] (or [] = all) -> [{name, enabled, disabled_reason,
        run_status (pending|running|completed|failed), run_error, ingested,
        errors:[{file, reason}], skipped:[{file, status}], marked_for_deletion}]
      "models":[names] (or [] = all) -> [{model, status
        (unknown|checking|healthy|unhealthy), embedding_dim, error}]
      "health": any value -> {embedder, qdrant, postgres} booleans"""
    return await (await _get_manager()).status(filters=filters)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
