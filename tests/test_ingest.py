"""Integration tests for the ingestion pipeline: real PG + in-memory
Qdrant + a fake embedder. Exercises every dispatch path and the WAL resume."""

import os

import psycopg
import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.config import ResolvedSectionConfig
from hulibaza.embedding_client import EmbeddingError
from hulibaza.files import discover_files
from hulibaza.identity import content_hash_file
from hulibaza.ingest import Ingestor
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get(
    "HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza"
)
DIM = 8
CAPS = dict(size_cap_text=10_000, size_cap_other=1_000)


class FakeEmbedder:
    """Duck-typed embedder. Deterministic vectors; can fail on demand."""

    def __init__(self, dim=DIM):
        self.dim = dim
        self.embed_calls = 0
        self.fail = False
        self.fail_on_call: int | None = None

    async def get_embedding_dim(self, model):
        return self.dim

    async def embed(self, model, texts):
        self.embed_calls += 1
        if self.fail or self.embed_calls == self.fail_on_call:
            raise EmbeddingError("boom")
        return [[float(len(t) % 10)] * self.dim for t in texts]


@pytest.fixture
async def state():
    store = StateStore(PG_URL)
    try:
        conn = await psycopg.AsyncConnection.connect(PG_URL, autocommit=True, connect_timeout=3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not available: {e}")
    async with conn:
        await conn.execute("DROP TABLE IF EXISTS file, section CASCADE")
    await store.init_schema()
    return store


@pytest.fixture
async def qdrant():
    store = QdrantStore(client=AsyncQdrantClient(location=":memory:"))
    yield store
    await store.aclose()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def ingestor(state, qdrant, embedder):
    return Ingestor(state, qdrant, embedder, deletion_grace_days=7)


def make_section(tmp_path, name="docs", chunk_size=50, batch=2, files=None):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    for rel, content in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return ResolvedSectionConfig(
        name=name, path=root, description="", embed_model="m",
        chunk_size=chunk_size, chunk_overlap=0, chunk_overlap_ratio=0.0,
        headroom_ratio=0.02, embed_batch_size=batch,
    )


async def _ingest(ingestor, section):
    return await ingestor.ingest_section(section, token_counter=len, **CAPS)


# ── new / unchanged ──


async def test_new_files_ingested(ingestor, state, qdrant, tmp_path):
    section = make_section(tmp_path, files={"a.md": "hello world alpha", "b.md": "beta gamma delta"})
    s = await _ingest(ingestor, section)
    assert s.status == "completed" and s.new == 2 and s.errored == 0
    for rel in ("a.md", "b.md"):
        f = await state.get_file("docs", rel)
        assert f.status == "ingested" and f.chunks_done == f.total_chunks and f.in_use
    assert await qdrant.count("docs") >= 2


async def test_unchanged_skipped_no_reembed(ingestor, state, embedder, tmp_path):
    section = make_section(tmp_path, files={"a.md": "hello world alpha beta"})
    await _ingest(ingestor, section)
    calls = embedder.embed_calls
    s = await _ingest(ingestor, section)
    assert s.unchanged == 1 and s.new == 0
    assert embedder.embed_calls == calls  # no re-embed


# ── changed = delete old + ingest new ──


async def test_changed_file_replaces_old(ingestor, state, qdrant, tmp_path):
    section = make_section(tmp_path, files={"a.md": "original content here"})
    await _ingest(ingestor, section)
    h1 = (await state.get_file("docs", "a.md")).content_hash
    count1 = await qdrant.count("docs")

    (section.path / "a.md").write_text("completely different replacement text now", encoding="utf-8")
    s = await _ingest(ingestor, section)
    assert s.changed == 1
    f = await state.get_file("docs", "a.md")
    assert f.content_hash != h1 and f.status == "ingested"
    # No orphan leak: only the new version's points remain for this file.
    files = {fi.source_file: fi for fi in await qdrant.list_files("docs")}
    assert set(files) == {"a.md"}
    assert files["a.md"].total_chunks == f.total_chunks


# ── resume (WAL) ──


async def test_resume_from_checkpoint(ingestor, state, qdrant, embedder, tmp_path):
    section = make_section(tmp_path, chunk_size=30, batch=2, files={"big.md": "word " * 200})
    cand = next(c for c in discover_files(section.path, **CAPS).files if c.rel_path == "big.md")
    chunks = ingestor._parse_and_chunk(section, cand, len)
    total = len(chunks)
    assert total > 2  # need multiple batches
    h = content_hash_file(cand.path)
    stat = cand.path.stat()

    # _store is called directly here (bypassing ingest_section), so create the
    # parent section row + collection the way the real flow would.
    await state.ensure_section("docs", "m", section.chunk_size, section.chunk_overlap)
    await qdrant.ensure_collection("docs", DIM)

    # Simulate a crash: fail on the 2nd embed call, after batch 1 (2 chunks) commits.
    embedder.fail_on_call = 2
    with pytest.raises(EmbeddingError):
        await ingestor._store(section, cand, h, stat, chunks, fresh=True, resume_from=0)
    f = await state.get_file("docs", "big.md")
    assert f.status == "ingesting" and f.chunks_done == 2
    assert await qdrant.count("docs") == 2  # only committed batch present

    # Resume with a healthy embedder — must continue from chunks_done, not restart.
    embedder.fail_on_call = None
    embedder.embed_calls = 0
    s = await _ingest(ingestor, section)
    assert s.resumed == 1
    f = await state.get_file("docs", "big.md")
    assert f.status == "ingested" and f.chunks_done == total
    assert await qdrant.count("docs") == total
    # Resume re-embedded only the remaining chunks (fewer calls than a full run).
    assert embedder.embed_calls < (total + 1) // 2 + 1


# ── move / reappear ──


async def test_move_repaths_without_reembed(ingestor, state, qdrant, embedder, tmp_path):
    section = make_section(tmp_path, files={"old/name.md": "movable content stays same"})
    await _ingest(ingestor, section)
    calls = embedder.embed_calls
    total = (await state.get_file("docs", "old/name.md")).total_chunks

    # Rename the file (same bytes) and re-ingest.
    (section.path / "old" / "name.md").rename(section.path / "renamed.md")
    s = await _ingest(ingestor, section)
    assert s.moved == 1 and s.new == 0
    assert embedder.embed_calls == calls  # reused vectors, no re-embed
    assert await state.get_file("docs", "old/name.md") is None
    moved = await state.get_file("docs", "renamed.md")
    assert moved.in_use and moved.total_chunks == total
    files = {fi.source_file for fi in await qdrant.list_files("docs")}
    assert files == {"renamed.md"}


async def test_reappeared_same_path_restored(ingestor, state, qdrant, embedder, tmp_path):
    section = make_section(tmp_path, files={"a.md": "content that comes back"})
    await _ingest(ingestor, section)
    await state.tombstone_file("docs", "a.md", grace_days=7)
    await qdrant.set_in_use("docs", "a.md", False)
    calls = embedder.embed_calls

    s = await _ingest(ingestor, section)  # same content, was tombstoned
    assert s.restored == 1
    f = await state.get_file("docs", "a.md")
    assert f.in_use and f.delete_after is None
    assert embedder.embed_calls == calls  # no re-embed


# ── fail-soft ──


async def test_fail_soft_marks_error_and_continues(state, qdrant, tmp_path):
    emb = FakeEmbedder()
    emb.fail = True  # every content embed fails; get_embedding_dim still works
    ing = Ingestor(state, qdrant, emb, deletion_grace_days=7)
    section = make_section(tmp_path, files={"a.md": "some text"})
    s = await ing.ingest_section(section, token_counter=len, **CAPS)
    assert s.status == "completed" and s.errored == 1  # run survives
    f = await state.get_file("docs", "a.md")
    assert f.status == "error" and f.error_reason


async def test_preflight_embedder_down_fails_run(state, qdrant, tmp_path):
    class DeadEmbedder(FakeEmbedder):
        async def get_embedding_dim(self, model):
            raise EmbeddingError("embedder unreachable")

    ing = Ingestor(state, qdrant, DeadEmbedder(), deletion_grace_days=7)
    section = make_section(tmp_path, files={"a.md": "text"})
    s = await ing.ingest_section(section, token_counter=len, **CAPS)
    assert s.status == "failed" and "unreachable" in s.reason


# ── skips / reset ──


async def test_skips_recorded(ingestor, state, tmp_path):
    section = make_section(tmp_path, files={
        "ok.md": "fine text",
        "big.unknownext": "x" * 2000,  # over the "other" cap (1000)
    })
    (section.path / "img.customext").write_bytes(b"GIF\x00\x00binary")  # null-byte binary
    s = await _ingest(ingestor, section)
    assert s.skipped == 2
    assert (await state.get_file("docs", "big.unknownext")).status == "skipped_size"
    assert (await state.get_file("docs", "img.customext")).status == "skipped_binary"


async def test_reset_on_param_change(ingestor, state, qdrant, tmp_path):
    section = make_section(tmp_path, chunk_size=50, files={"a.md": "some content for chunks"})
    await _ingest(ingestor, section)
    # Change an embedding parameter -> full reset (drop + re-ingest).
    changed = section.model_copy(update={"chunk_size": 30})
    s = await _ingest(ingestor, changed)
    assert s.reset is True and s.new == 1
    fp = await state.get_section("docs")
    assert fp.chunk_size == 30


async def test_already_running_when_locked(ingestor, state, tmp_path):
    section = make_section(tmp_path, files={"a.md": "text"})
    async with state.try_section_lock("docs") as got:
        assert got is True
        s = await _ingest(ingestor, section)  # lock held elsewhere
        assert s.status == "already_running"
