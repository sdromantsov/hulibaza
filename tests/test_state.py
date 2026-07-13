"""Integration tests for the Postgres state layer, against a real PG.

Uses the MVP compose Postgres (localhost:59942) by default; override with
HULIBAZA_TEST_PG. Skips cleanly when no Postgres is reachable.
"""

import os

import psycopg
import pytest

from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get(
    "HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza"
)


@pytest.fixture
async def state():
    store = StateStore(PG_URL)
    try:
        conn = await psycopg.AsyncConnection.connect(PG_URL, autocommit=True, connect_timeout=3)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Postgres not available: {e}")
    async with conn:
        await conn.execute("DROP TABLE IF EXISTS file, section CASCADE")
    await store.init_schema()
    return store


async def _seed_section(state, name="docs"):
    await state.ensure_section(name, "m", 512, 51)


# ── section fingerprint ──


async def test_init_schema_idempotent(state):
    await state.init_schema()  # second call must not raise
    assert await state.list_sections() == []


async def test_ensure_and_get_section(state):
    await state.ensure_section("docs", "nomic", 500, 60)
    fp = await state.get_section("docs")
    assert fp.embed_model == "nomic"
    assert fp.chunk_size == 500
    assert fp.chunk_overlap == 60
    assert fp.sizing_mode == "tokens"
    assert fp.sparse_format == "blake2b-idf"
    assert fp.ingested_at is None


async def test_mark_section_ingested(state):
    await _seed_section(state)
    await state.mark_section_ingested("docs")
    assert (await state.get_section("docs")).ingested_at is not None


async def test_ensure_section_preserves_ingested_at(state):
    await _seed_section(state)
    await state.mark_section_ingested("docs")
    before = (await state.get_section("docs")).ingested_at
    await state.ensure_section("docs", "m", 256, 20)  # config change
    fp = await state.get_section("docs")
    assert fp.chunk_size == 256
    assert fp.ingested_at == before  # not reset by a fingerprint update


# ── file lifecycle / resume ──


async def test_begin_sets_wal_intent(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 1234, 1000.0, total_chunks=10)
    f = await state.get_file("docs", "a.md")
    assert f.status == "ingesting"
    assert f.chunks_done == 0
    assert f.total_chunks == 10
    assert f.content_hash == "h" * 64
    assert f.size == 1234
    assert f.in_use is True


async def test_checkpoint_then_complete(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 10, 1.0, total_chunks=10)
    await state.set_chunks_done("docs", "a.md", 4)  # crash point after 4 chunks
    f = await state.get_file("docs", "a.md")
    assert f.chunks_done == 4 and f.status == "ingesting"  # resume from here
    await state.complete_file_ingest("docs", "a.md")
    f = await state.get_file("docs", "a.md")
    assert f.status == "ingested" and f.chunks_done == 10


async def test_reingest_resets_checkpoint(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "1" * 64, 10, 1.0, total_chunks=10)
    await state.complete_file_ingest("docs", "a.md")
    # Changed content: begin again with a new hash resets chunks_done.
    await state.begin_file_ingest("docs", "a.md", "2" * 64, 20, 2.0, total_chunks=5)
    f = await state.get_file("docs", "a.md")
    assert f.content_hash == "2" * 64
    assert f.status == "ingesting" and f.chunks_done == 0 and f.total_chunks == 5


async def test_mark_error(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 10, 1.0, total_chunks=3)
    await state.mark_file_error("docs", "a.md", "embedder 500")
    f = await state.get_file("docs", "a.md")
    assert f.status == "error" and f.error_reason == "embedder 500"


async def test_record_skip(state):
    await _seed_section(state)
    await state.record_skip("docs", "big.log", "skipped_size", "too_large:2000000>1000000", size=2_000_000)
    f = await state.get_file("docs", "big.log")
    assert f.status == "skipped_size" and f.in_use is False and f.size == 2_000_000
    with pytest.raises(ValueError):
        await state.record_skip("docs", "x", "ingested", "bad status")


async def test_list_files_sorted(state):
    await _seed_section(state)
    for name in ["c.md", "a.md", "b.md"]:
        await state.begin_file_ingest("docs", name, "h" * 64, 1, 1.0, total_chunks=1)
    assert [f.rel_path for f in await state.list_files("docs")] == ["a.md", "b.md", "c.md"]


# ── tombstone / move ──


async def test_tombstone_and_restore(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 1, 1.0, total_chunks=1)
    await state.complete_file_ingest("docs", "a.md")
    await state.tombstone_file("docs", "a.md", grace_days=7)
    f = await state.get_file("docs", "a.md")
    assert f.in_use is False and f.delete_after is not None
    await state.restore_file("docs", "a.md")
    f = await state.get_file("docs", "a.md")
    assert f.in_use is True and f.delete_after is None


async def test_find_by_hash_for_move(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "old/name.md", "SAME" * 16, 10, 1.0, total_chunks=2)
    await state.complete_file_ingest("docs", "old/name.md")
    matches = await state.find_by_hash("docs", "SAME" * 16)
    assert [m.rel_path for m in matches] == ["old/name.md"]
    await state.repath_file("docs", "old/name.md", "new/name.md")
    assert await state.get_file("docs", "old/name.md") is None
    moved = await state.get_file("docs", "new/name.md")
    assert moved.content_hash == "SAME" * 16 and moved.in_use is True


# ── reconcile / purge ──


async def test_list_ingesting_orphans(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "stuck.md", "h" * 64, 1, 1.0, total_chunks=5)
    await state.begin_file_ingest("docs", "done.md", "h" * 64, 1, 1.0, total_chunks=5)
    await state.complete_file_ingest("docs", "done.md")
    orphans = await state.list_ingesting("docs")
    assert [f.rel_path for f in orphans] == ["stuck.md"]


async def test_expired_tombstones(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "expired.md", "h" * 64, 1, 1.0, total_chunks=1)
    await state.complete_file_ingest("docs", "expired.md")
    await state.begin_file_ingest("docs", "fresh.md", "h" * 64, 1, 1.0, total_chunks=1)
    await state.complete_file_ingest("docs", "fresh.md")
    await state.tombstone_file("docs", "expired.md", grace_days=0)  # already past
    await state.tombstone_file("docs", "fresh.md", grace_days=7)  # still in grace
    expired = await state.list_expired_tombstones("docs")
    assert [f.rel_path for f in expired] == ["expired.md"]


async def test_delete_file_row(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 1, 1.0, total_chunks=1)
    await state.delete_file_row("docs", "a.md")
    assert await state.get_file("docs", "a.md") is None


async def test_delete_section_cascades(state):
    await _seed_section(state)
    await state.begin_file_ingest("docs", "a.md", "h" * 64, 1, 1.0, total_chunks=1)
    await state.delete_section("docs")
    assert await state.get_section("docs") is None
    assert await state.list_files("docs") == []  # cascaded


# ── advisory lock ──


async def test_advisory_lock_mutual_exclusion(state):
    async with state.try_section_lock("docs") as got_first:
        assert got_first is True
        # A second holder (separate session) cannot acquire it concurrently.
        async with state.try_section_lock("docs") as got_second:
            assert got_second is False
    # Released now → re-acquirable.
    async with state.try_section_lock("docs") as got_again:
        assert got_again is True


async def test_advisory_lock_distinct_sections(state):
    async with state.try_section_lock("s1") as a:
        async with state.try_section_lock("s2") as b:
            assert a is True and b is True  # different keys, no contention


async def test_health_check(state):
    assert await state.health_check() is True
