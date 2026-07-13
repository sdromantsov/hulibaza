"""Unit tests for the model-health registry. No infra."""

import pytest

from hulibaza.embedding_client import EmbeddingError
from hulibaza.health import ModelHealthRegistry

pytestmark = pytest.mark.unit


class FakeEmbedder:
    def __init__(self):
        self.fail = False
        self.dim = 8

    async def get_embedding_dim(self, model):
        if self.fail:
            raise EmbeddingError("embedder down")
        return self.dim


async def test_unknown_model_is_optimistically_usable():
    reg = ModelHealthRegistry(FakeEmbedder())
    assert reg.status("m") == "unknown"
    assert reg.is_usable("m") is True


async def test_check_healthy():
    reg = ModelHealthRegistry(FakeEmbedder())
    h = await reg.check("m")
    assert h.status == "healthy" and h.embedding_dim == 8
    assert reg.is_usable("m") is True


async def test_check_unhealthy():
    emb = FakeEmbedder()
    emb.fail = True
    reg = ModelHealthRegistry(emb)
    h = await reg.check("m")
    assert h.status == "unhealthy" and "down" in h.error
    assert reg.is_usable("m") is False


async def test_mark_and_recover():
    emb = FakeEmbedder()
    reg = ModelHealthRegistry(emb)
    reg.mark_unhealthy("m", "boom")
    assert reg.is_usable("m") is False
    # Embedder recovers; a recheck restores health.
    await reg.check("m")
    assert reg.status("m") == "healthy"


async def test_check_all_and_snapshot():
    emb = FakeEmbedder()
    reg = ModelHealthRegistry(emb)
    await reg.check_all(["a", "b"])
    snap = reg.snapshot()
    assert [s["model"] for s in snap] == ["a", "b"]
    assert all(s["status"] == "healthy" for s in snap)
    assert reg.snapshot(["a"]) == [s for s in snap if s["model"] == "a"]
