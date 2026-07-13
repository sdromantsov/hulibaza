"""Integration tests for the lifecycle daemon: real PG + in-memory
Qdrant + fake embedder. Verifies mark-only detection of change/delete/move."""

import os

import psycopg
import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.config import GlobalConfig, ResolvedSectionConfig
from hulibaza.daemon import Daemon
from hulibaza.embedding_client import EmbeddingError
from hulibaza.ingest import Ingestor
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza")
DIM = 8
CAPS = dict(size_cap_text=10_000, size_cap_other=1_000)


class FakeEmbedder:
    def __init__(self):
        self.embed_calls = 0

    async def get_embedding_dim(self, model):
        return DIM

    async def embed(self, model, texts):
        self.embed_calls += 1
        return [[float(len(t) % 5)] * DIM for t in texts]


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
    embedder = FakeEmbedder()
    ingestor = Ingestor(state, qdrant, embedder, deletion_grace_days=7)
    daemon = Daemon(GlobalConfig(), state, qdrant)
    yield state, qdrant, embedder, ingestor, daemon, tmp_path
    await qdrant.aclose()


def make_section(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return ResolvedSectionConfig(
        name="docs", path=root, description="", embed_model="m",
        chunk_size=50, chunk_overlap=0, chunk_overlap_ratio=0.0,
        headroom_ratio=0.02, embed_batch_size=8,
    )


async def _ingest(ingestor, section):
    return await ingestor.ingest_section(section, token_counter=len, **CAPS)


async def test_deleted_file_tombstoned_with_grace(env):
    state, qdrant, embedder, ingestor, daemon, tmp = env
    section = make_section(tmp / "docs", {"a.md": "alpha content", "b.md": "beta content"})
    await _ingest(ingestor, section)

    (section.path / "b.md").unlink()  # delete on disk
    await daemon.poll_section(section)

    row = await state.get_file("docs", "b.md")
    assert row.in_use is False and row.delete_after is not None  # grace set
    # Search no longer returns it.
    files = {f.source_file for f in await qdrant.list_files("docs")}
    assert files == {"a.md"}


async def test_changed_file_marked_no_grace(env):
    state, qdrant, embedder, ingestor, daemon, tmp = env
    section = make_section(tmp / "docs", {"a.md": "original short"})
    await _ingest(ingestor, section)

    (section.path / "a.md").write_text("a much longer and completely different body of text now")
    await daemon.poll_section(section)

    row = await state.get_file("docs", "a.md")
    assert row.in_use is False and row.delete_after is None  # changed => no grace (rebuilt by ingest)
    assert (await qdrant.list_files("docs")) == []  # tombstoned in Qdrant too


async def test_moved_file_repathed_no_reembed(env):
    state, qdrant, embedder, ingestor, daemon, tmp = env
    section = make_section(tmp / "docs", {"old/x.md": "portable movable content"})
    await _ingest(ingestor, section)
    calls = embedder.embed_calls
    total = (await state.get_file("docs", "old/x.md")).total_chunks

    (section.path / "old" / "x.md").rename(section.path / "new.md")
    await daemon.poll_section(section)

    assert await state.get_file("docs", "old/x.md") is None
    moved = await state.get_file("docs", "new.md")
    assert moved.in_use and moved.total_chunks == total
    assert embedder.embed_calls == calls  # reused vectors
    assert {f.source_file for f in await qdrant.list_files("docs")} == {"new.md"}


async def test_unchanged_no_spurious_marks(env):
    state, qdrant, embedder, ingestor, daemon, tmp = env
    section = make_section(tmp / "docs", {"a.md": "steady content", "b.md": "also steady"})
    await _ingest(ingestor, section)
    await daemon.poll_section(section)  # nothing changed on disk
    for rel in ("a.md", "b.md"):
        assert (await state.get_file("docs", rel)).in_use is True


async def test_daemon_skips_locked_section(env):
    state, qdrant, embedder, ingestor, daemon, tmp = env
    section = make_section(tmp / "docs", {"a.md": "content", "b.md": "content two"})
    await _ingest(ingestor, section)
    (section.path / "b.md").unlink()

    async with state.try_section_lock("docs") as got:  # simulate mid-ingest/purge
        assert got
        await daemon.poll_section(section)  # must skip, not act
    # b.md still in_use (daemon didn't touch the locked section).
    assert (await state.get_file("docs", "b.md")).in_use is True
