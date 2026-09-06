"""Unit tests for the async rerank client, driven by httpx.MockTransport."""

import json

import httpx
import pytest

from hulibaza.rerank_client import RerankClient, RerankError

pytestmark = pytest.mark.unit


def _client(handler, **kw) -> RerankClient:
    transport = httpx.MockTransport(handler)
    return RerankClient(
        "http://rerank.local",
        client=httpx.AsyncClient(transport=transport),
        backoff_factor=0.0,  # no real sleeping in tests
        **kw,
    )


def _rerank_response(pairs):
    return httpx.Response(200, json={
        "results": [{"index": i, "relevance_score": s} for i, s in pairs],
        "usage": {"total_tokens": 10},
    })


async def test_rerank_returns_pairs_sorted_desc():
    def handler(req):
        assert req.url.path == "/v1/rerank"
        body = json.loads(req.content)
        assert body["model"] == "rr" and body["query"] == "q"
        assert body["documents"] == ["a", "b", "c"]
        assert body["top_n"] == 3  # None -> all documents
        return _rerank_response([(2, 0.1), (0, 0.9), (1, 0.5)])

    async with _client(handler) as c:
        out = await c.rerank("rr", "q", ["a", "b", "c"])
    assert out == [(0, 0.9), (1, 0.5), (2, 0.1)]


async def test_rerank_empty_short_circuits():
    def handler(req):
        raise AssertionError("must not hit network for empty input")

    async with _client(handler) as c:
        assert await c.rerank("rr", "q", []) == []


async def test_rerank_top_n_truncates_after_sort():
    def handler(req):
        return _rerank_response([(0, 0.9), (1, 0.5), (2, 0.1)])

    async with _client(handler) as c:
        out = await c.rerank("rr", "q", ["a", "b", "c"], top_n=2)
    assert out == [(0, 0.9), (1, 0.5)]


async def test_rerank_missing_results_key_raises():
    def handler(req):
        return httpx.Response(200, json={"nope": []})

    async with _client(handler) as c:
        with pytest.raises(RerankError, match="no 'results'"):
            await c.rerank("rr", "q", ["a"])


async def test_rerank_http_error_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    async with _client(handler, max_retries=3) as c:
        with pytest.raises(RerankError, match="400"):
            await c.rerank("rr", "q", ["a"])
    assert calls["n"] == 1


async def test_rerank_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="warming up")
        return _rerank_response([(0, 0.7)])

    async with _client(handler, max_retries=3) as c:
        out = await c.rerank("rr", "q", ["a"])
    assert out == [(0, 0.7)]
    assert calls["n"] == 2


async def test_rerank_retries_exhausted_raises():
    def handler(req):
        return httpx.Response(503, text="still down")

    async with _client(handler, max_retries=2) as c:
        with pytest.raises(RerankError, match="503"):
            await c.rerank("rr", "q", ["a"])


async def test_rerank_connect_error_retried_then_raises():
    def handler(req):
        raise httpx.ConnectError("refused")

    async with _client(handler, max_retries=1) as c:
        with pytest.raises(RerankError, match="Cannot connect"):
            await c.rerank("rr", "q", ["a"])


async def test_rerank_timeout_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow")

    async with _client(handler, max_retries=3) as c:
        with pytest.raises(RerankError, match="timed out"):
            await c.rerank("rr", "q", ["a"])
    assert calls["n"] == 1  # no retry on timeout


async def test_is_model_available():
    def handler(req):
        return httpx.Response(200, json={"data": [{"id": "rr"}, {"id": "other"}]})

    async with _client(handler) as c:
        assert await c.is_model_available("rr") is True
        assert await c.is_model_available("ghost") is False


async def test_is_model_available_false_on_error():
    def handler(req):
        return httpx.Response(500)

    async with _client(handler) as c:
        assert await c.is_model_available("rr") is False


async def test_health_check():
    async with _client(lambda req: httpx.Response(200, json={"data": []})) as c:
        assert await c.health_check() is True

    async with _client(lambda req: httpx.Response(500)) as c:
        assert await c.health_check() is False
