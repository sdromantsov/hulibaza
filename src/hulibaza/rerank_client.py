"""Async rerank client over an OpenAI-compatible /v1/rerank endpoint.

Talks to any server implementing the Jina-format /v1/rerank (llama.cpp
llama-server in reranking mode, TEI, Jina AI). Takes a query + documents and
returns (index, relevance_score) pairs sorted by score descending. Reranking
is query-time only: the caller treats any failure as a degradation, never as
a search error.

POSTs are retried on transient connection or 5xx errors with exponential
backoff; timeouts are NOT retried.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600
_RETRYABLE_STATUS = frozenset({502, 503, 504})
_POOL = 16  # rerank is serial-GPU per search; a few spares for health pings


class RerankError(Exception):
    pass


class RerankClient:
    def __init__(
        self,
        base_url: str,
        timeout: int = DEFAULT_TIMEOUT,
        *,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=_POOL, max_keepalive_connections=_POOL),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "RerankClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post_with_retry(self, url: str, payload: dict) -> httpx.Response:
        last_error: RerankError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException:
                raise RerankError(f"Rerank request timed out after {self.timeout}s")
            except httpx.TransportError as e:
                last_error = RerankError(f"Cannot connect to {self.base_url}: {e}")
            else:
                if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = RerankError(f"HTTP {resp.status_code}: {resp.text}")
                else:
                    return resp
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_factor * (2 ** attempt))
        raise last_error or RerankError("rerank request failed")

    async def rerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Score documents against the query.

        Returns [(original_index, relevance_score), ...] sorted by score
        descending. top_n=None scores all documents.
        """
        if not documents:
            return []

        payload: dict = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
        }
        resp = await self._post_with_retry(f"{self.base_url}/v1/rerank", payload)
        if resp.status_code >= 400:
            raise RerankError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        results = data.get("results")
        if results is None:
            raise RerankError(f"Unexpected response (no 'results' key): {data}")

        scored = [
            (int(r["index"]), float(r["relevance_score"]))
            for r in results
            if "index" in r and "relevance_score" in r
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            scored = scored[:top_n]
        return scored

    async def is_model_available(self, model: str) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            return model in {m.get("id") for m in entries}
        except Exception as e:
            logger.warning("Failed to list models for reranker check: %s", e)
            return False

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
