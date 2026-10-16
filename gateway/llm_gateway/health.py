"""Periodic per-backend health checks."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass

import httpx

from .config import BackendConfig, GatewayConfig

log = logging.getLogger("llm_gateway.health")


@dataclass
class BackendHealth:
    healthy: bool | None = None
    checked_at: float | None = None
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class HealthChecker:
    def __init__(self, config: GatewayConfig, client: httpx.AsyncClient) -> None:
        self._config = config
        self._client = client
        self.state: dict[str, BackendHealth] = {name: BackendHealth() for name in config.backends}

    def snapshot(self) -> dict[str, bool | None]:
        return {name: h.healthy for name, h in self.state.items()}

    async def check_one(self, backend: BackendConfig) -> BackendHealth:
        start = time.perf_counter()
        headers = {"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else None
        try:
            resp = await self._client.get(
                backend.health_url, headers=headers, timeout=self._config.health_timeout_s
            )
            healthy = resp.is_success
            error = None if healthy else f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            healthy, error = False, f"{type(exc).__name__}: {exc}"
        result = BackendHealth(
            healthy=healthy,
            checked_at=time.time(),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=error,
        )
        previous = self.state.get(backend.name)
        if previous is None or previous.healthy != healthy:
            log.info(
                "backend health changed",
                extra={"ctx": {"backend": backend.name, "healthy": healthy, "error": error}},
            )
        self.state[backend.name] = result
        return result

    async def check_all(self) -> dict[str, BackendHealth]:
        await asyncio.gather(*(self.check_one(b) for b in self._config.backends.values()))
        return self.state

    async def run_forever(self) -> None:
        while True:
            try:
                await self.check_all()
            except Exception:  # never let the loop die
                log.exception("health check loop failed")
            await asyncio.sleep(self._config.health_interval_s)
