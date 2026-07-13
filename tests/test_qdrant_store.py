"""Integration-style tests for the async Qdrant store, run against the real
local engine via AsyncQdrantClient(location=":memory:") — no server needed."""

import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.identity import point_id
from hulibaza.qdrant_store import ChunkPoint, QdrantStore
from hulibaza.sparse import build_sparse_vector

pytestmark = pytest.mark.unit  # in-memory engine, no external service

COLL = "docs"
DIM = 4


@pytest.fixture
async def store():
    s = QdrantStore(client=AsyncQdrantClient(location=":memory:"))
    await s.ensure_collection(COLL, DIM)
    yield s
    await s.aclose()


def _chunk(source_file, chunk_index, content_hash="h" * 64, dense=None, text=None, in_use=True):
    text = text or f"{source_file} chunk {chunk_index} alpha beta"
    return ChunkPoint(
        id=point_id("docs", source_file, content_hash, chunk_index),
        text=text,
        dense=dense or [0.1 * (chunk_index + 1)] * DIM,
        sparse=build_sparse_vector(text),
        source_file=source_file,
        page_number=chunk_index // 2,
        chunk_index=chunk_index,
        section_name="docs",
        in_use=in_use,
    )


async def test_ensure_collection_idempotent(store):
    await store.ensure_collection(COLL, DIM)  # second call must be a no-op
    assert await store.collection_exists(COLL)


async def test_upsert_and_count(store):
    n = await store.upsert_chunks(COLL, [_chunk("a.md", 0), _chunk("a.md", 1)])
    assert n == 2
    assert await store.count(COLL) == 2


async def test_deterministic_upsert_is_idempotent(store):
    # Same (section, file, hash, index) -> same point id -> overwrite, no dup.
    await store.upsert_chunks(COLL, [_chunk("a.md", 0)])
    await store.upsert_chunks(COLL, [_chunk("a.md", 0)])
    assert await store.count(COLL) == 1


async def test_changed_content_coexists(store):
    # Different content_hash -> different id -> old and new both stored.
    await store.upsert_chunks(COLL, [_chunk("a.md", 0, content_hash="1" * 64)])
    await store.upsert_chunks(COLL, [_chunk("a.md", 0, content_hash="2" * 64)])
    assert await store.count(COLL) == 2


async def test_semantic_search(store):
    await store.upsert_chunks(COLL, [
        _chunk("a.md", 0, dense=[1.0, 0.0, 0.0, 0.0]),
        _chunk("b.md", 0, dense=[0.0, 1.0, 0.0, 0.0]),
    ])
    hits = await store.semantic_search(COLL, [1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(hits) == 1
    assert hits[0].source_file == "a.md"


async def test_keyword_search(store):
    await store.upsert_chunks(COLL, [
        _chunk("a.md", 0, text="cudamalloc device memory allocate"),
        _chunk("b.md", 0, text="unrelated prose about weather"),
    ])
    hits = await store.keyword_search(COLL, build_sparse_vector("cudamalloc"), limit=3)
    assert hits and hits[0].source_file == "a.md"


async def test_hybrid_search_returns_results(store):
    await store.upsert_chunks(COLL, [
        _chunk("a.md", 0, dense=[1.0, 0.0, 0.0, 0.0], text="alpha keyword match here"),
        _chunk("b.md", 0, dense=[0.0, 0.0, 0.0, 1.0], text="beta something else"),
    ])
    hits = await store.hybrid_search(
        COLL, [1.0, 0.0, 0.0, 0.0], build_sparse_vector("alpha keyword"), limit=2
    )
    assert hits and hits[0].source_file == "a.md"


async def test_search_excludes_in_use_false(store):
    await store.upsert_chunks(COLL, [_chunk("a.md", 0, dense=[1.0, 0.0, 0.0, 0.0], in_use=False)])
    hits = await store.semantic_search(COLL, [1.0, 0.0, 0.0, 0.0], limit=5)
    assert hits == []


async def test_set_in_use_tombstones_from_search(store):
    await store.upsert_chunks(COLL, [_chunk("a.md", 0, dense=[1.0, 0.0, 0.0, 0.0])])
    assert len(await store.semantic_search(COLL, [1.0, 0.0, 0.0, 0.0], limit=5)) == 1
    await store.set_in_use(COLL, "a.md", False)
    assert await store.semantic_search(COLL, [1.0, 0.0, 0.0, 0.0], limit=5) == []


async def test_delete_by_file(store):
    await store.upsert_chunks(COLL, [_chunk("a.md", 0), _chunk("a.md", 1), _chunk("b.md", 0)])
    await store.delete_by_file(COLL, "a.md")
    assert await store.count(COLL) == 1
    files = await store.list_files(COLL)
    assert [f.source_file for f in files] == ["b.md"]


async def test_list_files_aggregates(store):
    await store.upsert_chunks(COLL, [
        _chunk("a.md", 0), _chunk("a.md", 1), _chunk("a.md", 2),
        _chunk("b.md", 0),
    ])
    files = {f.source_file: f for f in await store.list_files(COLL)}
    assert files["a.md"].total_chunks == 3
    assert files["a.md"].max_page == 1  # chunk_index 2 // 2
    assert files["b.md"].total_chunks == 1


async def test_get_chunks_by_index_ordered(store):
    await store.upsert_chunks(COLL, [_chunk("a.md", i) for i in range(5)])
    got = await store.get_chunks_by_index(COLL, "a.md", start=1, count=3)
    assert [c.chunk_index for c in got] == [1, 2, 3]
    assert await store.get_chunks_by_index(COLL, "a.md", start=0, count=0) == []


async def test_health_check(store):
    assert await store.health_check() is True
