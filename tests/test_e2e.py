"""End-to-end integration test on a REAL section (the provided uniswap-v4 wiki):
real files -> discover -> parse -> chunk -> (fake) embed -> Qdrant + PG -> search.

Uses a fake embedder + char-length token counter (no GPU/model needed); the goal
is to exercise the whole pipeline on real markdown + Python content. Read-only
with respect to the example directory."""

import os
from pathlib import Path

import psycopg
import pytest
from qdrant_client import AsyncQdrantClient

from hulibaza.config import Defaults, GlobalConfig, ModelSpec
from hulibaza.ingest import Ingestor
from hulibaza.manager import HulibazaManager
from hulibaza.qdrant_store import QdrantStore
from hulibaza.state import StateStore

pytestmark = pytest.mark.integration

PG_URL = os.environ.get("HULIBAZA_TEST_PG", "postgresql://hulibaza:hulibaza@localhost:59942/hulibaza")
EXAMPLE_WIKI = Path(os.environ.get(
    "HULIBAZA_EXAMPLE_WIKI", "/home/ahmdt/Desktop/clauding/hulibaza/test_wiki"
))
SECTION = "uniswap-v4"
DIM = 16
MODEL = "qwen3-embed-4b"  # what the example's section.yaml declares


class FakeEmbedder:
    async def get_embedding_dim(self, model):
        return DIM

    async def embed(self, model, texts):
        # Deterministic pseudo-embedding from the text — enough for RRF to run.
        return [[float((hash(t) >> (8 * i)) % 17) for i in range(DIM)] for t in texts]

    async def embed_single(self, model, text):
        return (await self.embed(model, [text]))[0]

    async def health_check(self):
        return True


@pytest.fixture
async def manager(tmp_path):
    if not (EXAMPLE_WIKI / SECTION / "section.yaml").exists():
        pytest.skip(f"example wiki not found at {EXAMPLE_WIKI}")
    try:
        conn = await psycopg.AsyncConnection.connect(PG_URL, autocommit=True, connect_timeout=3)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Postgres not available: {e}")
    async with conn:
        await conn.execute("DROP TABLE IF EXISTS file, section CASCADE")

    tok = tmp_path / "tok.json"
    tok.write_text("{}")
    config = GlobalConfig(
        wiki_dir=str(EXAMPLE_WIKI),
        models={MODEL: ModelSpec(max_context=2048, tokenizer_path=str(tok))},
        defaults=Defaults(embed_model=MODEL, chunk_size=512),
    )
    state = StateStore(PG_URL)
    await state.init_schema()
    qdrant = QdrantStore(client=AsyncQdrantClient(location=":memory:"))
    ingestor = Ingestor(state, qdrant, FakeEmbedder(), deletion_grace_days=7)
    mgr = HulibazaManager(config, FakeEmbedder(), qdrant, state, ingestor,
                          get_token_counter=lambda model: len)
    yield mgr
    await qdrant.aclose()


async def test_full_pipeline_on_real_section(manager):
    # The section is discovered from the real directory.
    names = {s["name"] for s in manager.sections()["sections"]}
    assert SECTION in names

    # Ingest it end to end.
    manager.ingest(SECTION)
    await manager._tasks[SECTION]

    details = await manager.section_details(SECTION)
    cov = details["coverage"]
    assert cov["on_disk"] >= 18  # 9 markdown + 9 python (section.yaml ignored)
    assert cov["ingested"] == cov["on_disk"]  # everything ingested
    assert cov["pending"] == 0 and cov["in_progress"] == 0

    # list_files reflects the real files.
    files = {f["source_file"] for f in (await manager.list_files(SECTION))["files"]}
    assert any(f.endswith(".md") for f in files)
    assert any(f.endswith(".py") for f in files)

    # Keyword search finds real content (model-independent, no embedder needed).
    res = await manager.search(SECTION, "swap", mode="keyword", top_k=5)
    assert res.get("results"), res
    assert all("source_file" in r for r in res["results"])

    # Hybrid search runs end to end (RRF over dense+sparse).
    hyb = await manager.search(SECTION, "pool manager", mode="hybrid", top_k=3)
    assert "results" in hyb

    # get_chunks reads a real file's head.
    a_file = sorted(files)[0]
    chunks = (await manager.get_chunks(SECTION, a_file, 0, 3))["chunks"]
    assert chunks and chunks[0]["chunk_index"] == 0
