"""Model-health registry.

Tracks per-model embedder health so search/ingest can pre-flight: a model that
is checking or unhealthy blocks dense modes (keyword still works). Models are
checked at startup; an embed failure at runtime flips a model unhealthy, and a
background loop rechecks unhealthy models so they recover automatically.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ModelHealth:
    model: str
    status: str  # unknown | checking | healthy | unhealthy
    embedding_dim: int | None = None
    error: str | None = None
    checked_at: float | None = None


class ModelHealthRegistry:
    def __init__(self, embedder, *, now: Callable[[], float] = time.time) -> None:
        self._embedder = embedder
        self._now = now
        self._health: dict[str, ModelHealth] = {}

    def status(self, model: str) -> str:
        h = self._health.get(model)
        return h.status if h else "unknown"

    def get(self, model: str) -> ModelHealth:
        return self._health.get(model, ModelHealth(model, "unknown"))

    def is_usable(self, model: str) -> bool:
        # Unknown = not yet checked -> allow an optimistic attempt.
        return self.status(model) in ("healthy", "unknown")

    def mark_healthy(self, model: str, dim: int) -> None:
        self._health[model] = ModelHealth(model, "healthy", dim, None, self._now())

    def mark_unhealthy(self, model: str, error: str) -> None:
        logger.warning("Model '%s' marked unhealthy: %s", model, error)
        self._health[model] = ModelHealth(model, "unhealthy", None, error, self._now())

    async def check(self, model: str) -> ModelHealth:
        self._health[model] = ModelHealth(model, "checking", None, None, self._now())
        try:
            dim = await self._embedder.get_embedding_dim(model)
            self.mark_healthy(model, dim)
        except Exception as e:
            self.mark_unhealthy(model, str(e))
        return self._health[model]

    async def check_all(self, models) -> None:
        for model in models:
            await self.check(model)

    def snapshot(self, names=None) -> list[dict]:
        items = self._health.values()
        if names:
            items = [h for h in items if h.model in names]
        return [
            {"model": h.model, "status": h.status, "embedding_dim": h.embedding_dim, "error": h.error}
            for h in sorted(items, key=lambda h: h.model)
        ]

    async def run_recheck_loop(self, interval_seconds: int, stop: asyncio.Event) -> None:
        """Periodically recheck models currently marked unhealthy."""
        while not stop.is_set():
            for model, h in list(self._health.items()):
                if h.status == "unhealthy":
                    await self.check(model)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass
