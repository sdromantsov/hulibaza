"""Async OpenAI-compatible embedding client.

Talks to any server implementing OpenAI's /v1/embeddings and /v1/models
(llama.cpp llama-server / llama-swap router, Ollama >= 0.2, vLLM, LM Studio).
Tokenization is NOT done here — token counts come from the local tokenizer.

Embedding is idempotent, so POSTs are safely retried on transient connection or
5xx errors with exponential backoff. Timeouts are NOT retried (an embedding call
can legitimately take minutes; a retry would just pile on).
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600
_RETRYABLE_STATUS = frozenset({502, 503, 504})
_POOL = 16  # matches the serial-GPU calling pattern; a few spare for health pings


class EmbeddingError(Exception):
    pass


class EmbeddingClient:
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
        # An injectable client lets tests drive an httpx.MockTransport with no
        # network. When we create it, we own its lifecycle (aclose).
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=_POOL, max_keepalive_connections=_POOL),
        )
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post_with_retry(self, url: str, payload: dict) -> httpx.Response:
        last_error: EmbeddingError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException:
                raise EmbeddingError(
                    f"Embedding request timed out after {self.timeout}s"
                )
            except httpx.TransportError as e:
                last_error = EmbeddingError(f"Cannot connect to {self.base_url}: {e}")
            else:
                if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = EmbeddingError(f"HTTP {resp.status_code}: {resp.text}")
                else:
                    return resp
            # Exponential backoff before the next attempt.
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_factor * (2 ** attempt))
        raise last_error or EmbeddingError("embedding request failed")

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        url = f"{self.base_url}/v1/embeddings"
        resp = await self._post_with_retry(url, {"model": model, "input": texts})
        if resp.status_code >= 400:
            raise EmbeddingError(f"HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        entries = data.get("data")
        if entries is None:
            raise EmbeddingError(f"Unexpected response (no 'data' key): {data}")
        if len(entries) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: sent {len(texts)}, got {len(entries)}"
            )

        # OpenAI returns an "index" per entry; restore request order.
        ordered = sorted(entries, key=lambda e: e.get("index", 0))
        return [e["embedding"] for e in ordered]

    async def embed_single(self, model: str, text: str) -> list[float]:
        vectors = await self.embed(model, [text])
        return vectors[0]

    async def get_embedding_dim(self, model: str) -> int:
        return len(await self.embed_single(model, "dimension test"))

    async def is_model_available(self, model: str) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=10)
            resp.raise_for_status()
            entries = resp.json().get("data", [])
            return model in {m.get("id") for m in entries}
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
            return False

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
