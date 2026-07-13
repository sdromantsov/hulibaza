"""Integration tests for the purge process: real PG + in-memory Qdrant."""

import os

import psycopg
import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.config import ResolvedSectionConfig
from hulibaza.ingest import Ingestor
from hulibaza.purge import run_purge
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza")
DIM = 8
CAPS = dict(size_cap_text=10_000, size_cap_other=1_000)


class FakeEmbedder:
    async def get_embedding_dim(self, model):
        return DIM

    async def embed(self, model, texts):
        return [[1.0] * DIM for _ in texts]


@pytest.fixture
async def env(tmp_path):
    try:
        conn = await psycopg.AsyncConnection.connect(PG_URL, autocommit=True, connect_timeout=3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not available: {e}")
    async with conn:
        await conn.execute("DROP TABLE IF EXISTS file, section CASCADE")
    state = StateStore(PG_URL)
    await state.init_schema()
    qdrant = QdrantStore(client=AsyncQdrantClient(location=":memory:"))
    ingestor = Ingestor(state, qdrant, FakeEmbedder(), deletion_grace_days=7)

    root = tmp_path / "docs"
    root.mkdir()
    for rel in ("a.md", "b.md"):
        (root / rel).write_text(f"content of {rel}")
    section = ResolvedSectionConfig(
        name="docs", path=root, description="", embed_model="m", chunk_size=50,
        chunk_overlap=0, chunk_overlap_ratio=0.0, headroom_ratio=0.02, embed_batch_size=8,
    )
    await ingestor.ingest_section(section, token_counter=len, **CAPS)
    yield state, qdrant
    await qdrant.aclose()


async def test_purge_removes_expired_only(env):
    state, qdrant = env
    await state.tombstone_file("docs", "a.md", grace_days=0)  # expired now
    await state.tombstone_file("docs", "b.md", grace_days=7)  # still in grace

    results = await run_purge(state, qdrant)
    assert results["docs"] == ["a.md"]
    assert await state.get_file("docs", "a.md") is None  # PG row gone
    assert await state.get_file("docs", "b.md") is not None  # untouched
    files = {f.source_file for f in await qdrant.list_files("docs")}
    assert "a.md" not in files  # Qdrant points gone


async def test_purge_nothing_when_no_expired(env):
    state, qdrant = env
    assert await run_purge(state, qdrant) == {}


async def test_purge_skips_locked_section(env):
    state, qdrant = env
    await state.tombstone_file("docs", "a.md", grace_days=0)
    async with state.try_section_lock("docs") as got:  # section mid-ingest
        assert got
        results = await run_purge(state, qdrant)
    assert results["docs"] == "skipped (locked)"
    assert await state.get_file("docs", "a.md") is not None  # not purged while locked
