"""Unit tests for the async embedding client, driven by httpx.MockTransport."""

import httpx
import pytest

from hulibaza.embedding_client import EmbeddingClient, EmbeddingError

pytestmark = pytest.mark.unit


def _client(handler, **kw) -> EmbeddingClient:
    transport = httpx.MockTransport(handler)
    return EmbeddingClient(
        "http://embed.local",
        client=httpx.AsyncClient(transport=transport),
        backoff_factor=0.0,  # no real sleeping in tests
        **kw,
    )


def _embeddings_response(vectors, indexed=True):
    data = []
    for i, v in enumerate(vectors):
        entry = {"embedding": v}
        if indexed:
            entry["index"] = i
        data.append(entry)
    return httpx.Response(200, json={"data": data})


async def test_embed_returns_vectors():
    def handler(req):
        assert req.url.path == "/v1/embeddings"
        return _embeddings_response([[1.0, 2.0], [3.0, 4.0]])

    async with _client(handler) as c:
        out = await c.embed("m", ["a", "b"])
    assert out == [[1.0, 2.0], [3.0, 4.0]]


async def test_embed_empty_short_circuits():
    def handler(req):
        raise AssertionError("must not hit network for empty input")

    async with _client(handler) as c:
        assert await c.embed("m", []) == []


async def test_embed_restores_index_order():
    def handler(req):
        # Server returns out of order; client must sort by index.
        return httpx.Response(200, json={"data": [
            {"embedding": [2.0], "index": 1},
            {"embedding": [1.0], "index": 0},
        ]})

    async with _client(handler) as c:
        out = await c.embed("m", ["first", "second"])
    assert out == [[1.0], [2.0]]


async def test_embed_count_mismatch_raises():
    def handler(req):
        return _embeddings_response([[1.0]])  # only one, but two requested

    async with _client(handler) as c:
        with pytest.raises(EmbeddingError, match="mismatch"):
            await c.embed("m", ["a", "b"])


async def test_embed_missing_data_key_raises():
    def handler(req):
        return httpx.Response(200, json={"nope": []})

    async with _client(handler) as c:
        with pytest.raises(EmbeddingError, match="no 'data'"):
            await c.embed("m", ["a"])


async def test_embed_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="warming up")
        return _embeddings_response([[9.0]])

    async with _client(handler, max_retries=3) as c:
        out = await c.embed("m", ["a"])
    assert out == [[9.0]]
    assert calls["n"] == 2


async def test_embed_retries_exhausted_raises():
    def handler(req):
        return httpx.Response(503, text="still down")

    async with _client(handler, max_retries=2) as c:
        with pytest.raises(EmbeddingError, match="503"):
            await c.embed("m", ["a"])


async def test_embed_connect_error_retried_then_raises():
    def handler(req):
        raise httpx.ConnectError("refused")

    async with _client(handler, max_retries=1) as c:
        with pytest.raises(EmbeddingError, match="Cannot connect"):
            await c.embed("m", ["a"])


async def test_embed_timeout_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow")

    async with _client(handler, max_retries=3) as c:
        with pytest.raises(EmbeddingError, match="timed out"):
            await c.embed("m", ["a"])
    assert calls["n"] == 1  # no retry on timeout


async def test_get_embedding_dim():
    def handler(req):
        return _embeddings_response([[0.1, 0.2, 0.3, 0.4]])

    async with _client(handler) as c:
        assert await c.get_embedding_dim("m") == 4


async def test_is_model_available():
    def handler(req):
        return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}]})

    async with _client(handler) as c:
        assert await c.is_model_available("m2") is True
        assert await c.is_model_available("ghost") is False


async def test_health_check():
    async with _client(lambda req: httpx.Response(200, json={"data": []})) as c:
        assert await c.health_check() is True

    async with _client(lambda req: httpx.Response(500)) as c:
        assert await c.health_check() is False
