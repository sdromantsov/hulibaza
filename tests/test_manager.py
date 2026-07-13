"""Integration tests for the serving manager: retrieval + both
gates + navigation + ingest registry + status. Real PG, in-memory Qdrant, fake
embedder."""

import os

import psycopg
import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.config import Defaults, GlobalConfig, ModelSpec
from hulibaza.embedding_client import EmbeddingError
from hulibaza.ingest import Ingestor
from hulibaza.manager import HulibazaManager
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza")
DIM = 8


class FakeEmbedder:
    def __init__(self):
        self.fail = False

    async def get_embedding_dim(self, model):
        return DIM

    async def embed(self, model, texts):
        if self.fail:
            raise EmbeddingError("down")
        return [[float(len(t) % 5)] * DIM for t in texts]

    async def embed_single(self, model, text):
        return (await self.embed(model, [text]))[0]

    async def health_check(self):
        return not self.fail


@pytest.fixture
async def clean_pg():
    try:
        conn = await psycopg.AsyncConnection.connect(PG_URL, autocommit=True, connect_timeout=3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not available: {e}")
    async with conn:
        await conn.execute("DROP TABLE IF EXISTS file, section CASCADE")


@pytest.fixture
async def env(clean_pg, tmp_path):
    # wiki_dir with one section "docs" (section.yaml + files) + a dummy tokenizer.
    tok = tmp_path / "tok.json"
    tok.write_text("{}")
    wiki = tmp_path / "wiki"
    docs = wiki / "docs"
    docs.mkdir(parents=True)
    (docs / "section.yaml").write_text("description: Docs\nchunk_size: 50\n")
    (docs / "cuda.md").write_text("cudamalloc allocates device memory on the gpu")
    (docs / "kin.md").write_text("inverse kinematics solver for robot arms")

    config = GlobalConfig(
        wiki_dir=str(wiki),
        models={"m": ModelSpec(max_context=2048, tokenizer_path=str(tok))},
        defaults=Defaults(embed_model="m", chunk_size=50),
    )
    state = StateStore(PG_URL)
    await state.init_schema()
    qdrant = QdrantStore(client=AsyncQdrantClient(location=":memory:"))
    embedder = FakeEmbedder()
    ingestor = Ingestor(state, qdrant, embedder, deletion_grace_days=7)
    mgr = HulibazaManager(config, embedder, qdrant, state, ingestor,
                          get_token_counter=lambda model: len)
    yield mgr, docs, state, embedder
    await qdrant.aclose()


async def _ingest(mgr):
    mgr.ingest("docs")
    await mgr._tasks["docs"]  # await the background task to completion


# ── discovery / ingest registry ──


async def test_sections_lists_enabled(env):
    mgr, *_ = env
    out = mgr.sections()
    assert out["sections"][0]["name"] == "docs" and out["sections"][0]["enabled"]


async def test_ingest_then_status_completed(env):
    mgr, *_ = env
    r = mgr.ingest("docs")
    assert r["status"] == "started"
    await mgr._tasks["docs"]
    st = await mgr.status({"sections": ["docs"]})
    sec = st["sections"][0]
    assert sec["run_status"] == "completed" and sec["ingested"] == 2


# ── retrieval ──


async def test_search_keyword(env):
    mgr, *_ = env
    await _ingest(mgr)
    r = await mgr.search("docs", "cudamalloc", mode="keyword")
    assert r["results"] and r["results"][0]["source_file"] == "cuda.md"


async def test_search_hybrid_returns_results(env):
    mgr, *_ = env
    await _ingest(mgr)
    r = await mgr.search("docs", "kinematics", mode="hybrid")
    assert "results" in r and len(r["results"]) >= 1


async def test_search_unknown_section(env):
    mgr, *_ = env
    assert "error" in await mgr.search("ghost", "x")


async def test_search_before_ingest_errors(env):
    mgr, *_ = env
    r = await mgr.search("docs", "x")
    assert "not been ingested" in r["error"]


# ── validity gate ──


async def test_validity_blocks_dense_allows_keyword(env):
    mgr, docs, state, _ = env
    await _ingest(mgr)
    # Corrupt the stored fingerprint so it mismatches the section config.
    await state.ensure_section("docs", "m", 999, 0)
    hybrid = await mgr.search("docs", "cudamalloc", mode="hybrid")
    assert "Validity gate" in hybrid["error"]
    kw = await mgr.search("docs", "cudamalloc", mode="keyword")
    assert kw["results"] and "warnings" in kw  # keyword survives, warns


# ── completeness gate ──


async def test_completeness_blocks_then_allow_incomplete(env):
    mgr, docs, *_ = env
    await _ingest(mgr)
    (docs / "new.md").write_text("a brand new unindexed file appears")  # pending
    blocked = await mgr.search("docs", "cudamalloc", mode="keyword")
    assert "not fully ingested" in blocked["error"]
    assert "new.md" in blocked["excluded"]["pending"]
    allowed = await mgr.search("docs", "cudamalloc", mode="keyword", allow_incomplete=True)
    assert allowed["results"] and "warnings" in allowed


# ── navigation ──


async def test_list_files_and_get_chunks(env):
    mgr, *_ = env
    await _ingest(mgr)
    lf = await mgr.list_files("docs")
    names = {f["source_file"] for f in lf["files"]}
    assert names == {"cuda.md", "kin.md"}
    gc = await mgr.get_chunks("docs", "cuda.md", 0, 10)
    assert gc["chunks"] and gc["chunks"][0]["chunk_index"] == 0


# ── section_details ──


async def test_section_details_coverage(env):
    mgr, docs, *_ = env
    await _ingest(mgr)
    (docs / "extra.md").write_text("pending file for coverage")
    d = await mgr.section_details("docs")
    assert d["coverage"]["ingested"] == 2 and d["coverage"]["pending"] == 1
    assert d["embed_model"] == "m"


# ── embedder-down path ──


async def test_dense_search_embedder_down_suggests_keyword(env):
    mgr, docs, state, embedder = env
    await _ingest(mgr)
    embedder.fail = True
    r = await mgr.search("docs", "cudamalloc", mode="semantic")
    assert "mode='keyword'" in r["error"]


async def test_ingest_all_cleans_orphan_sections(env):
    mgr, docs, state, _ = env
    # A section tracked in PG + Qdrant whose directory does not exist on disk.
    await state.ensure_section("ghost", "m", 50, 5)
    await mgr.qdrant.ensure_collection("ghost", 8)
    mgr.ingest("all")
    await mgr._orphan_task
    for t in mgr._tasks.values():
        await t
    assert await state.get_section("ghost") is None  # PG rows dropped
    assert not await mgr.qdrant.collection_exists("ghost")  # collection dropped


async def test_unhealthy_model_blocks_dense_preflight(env):
    mgr, docs, state, embedder = env
    await _ingest(mgr)
    mgr.health.mark_unhealthy("m", "embedder crashed")  # pre-flight
    hybrid = await mgr.search("docs", "cudamalloc", mode="hybrid")
    assert "unhealthy" in hybrid["error"] and "keyword" in hybrid["error"]
    kw = await mgr.search("docs", "cudamalloc", mode="keyword")
    assert kw["results"]  # keyword unaffected by model health
