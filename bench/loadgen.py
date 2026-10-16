"""Async closed-loop load generator for OpenAI-compatible streaming endpoints."""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .metrics import RequestResult


@dataclass
class LoadConfig:
    base_url: str
    model: str
    max_tokens: int = 128
    temperature: float = 0.0
    endpoint: str = "chat"  # "chat" or "completions"
    include_usage: bool = True
    ignore_eos: bool = False
    api_key: str | None = None
    timeout_s: float = 600.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def url(self) -> str:
        path = "chat/completions" if self.endpoint == "chat" else "completions"
        return f"{self.base_url.rstrip('/')}/{path}"

    def payload(self, prompt: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.endpoint == "chat":
            body["messages"] = [{"role": "user", "content": prompt}]
        else:
            body["prompt"] = prompt
        if self.include_usage:
            body["stream_options"] = {"include_usage": True}
        if self.ignore_eos:
            body["ignore_eos"] = True
        body.update(self.extra_body)
        return body

    def headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers


def _chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta")
    if isinstance(delta, dict):
        return delta.get("content") or delta.get("reasoning_content") or ""
    return choice.get("text") or ""


async def stream_one(
    client: httpx.AsyncClient, cfg: LoadConfig, prompt: str, concurrency: int, t0: float
) -> RequestResult:
    """Send one streaming request and time every content chunk."""
    result = RequestResult(concurrency=concurrency, ok=False)
    start = time.perf_counter()
    result.start_s = start - t0
    token_times: list[float] = []
    usage_tokens: int | None = None
    try:
        async with client.stream(
            "POST", cfg.url, json=cfg.payload(prompt), headers=cfg.headers()
        ) as resp:
            result.status = resp.status_code
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                result.error = f"HTTP {resp.status_code}: {body[:200]}"
                return result
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if "error" in chunk:
                    result.error = str(chunk["error"])[:200]
                    return result
                if _chunk_text(chunk):
                    token_times.append(time.perf_counter())
                usage = chunk.get("usage")
                if isinstance(usage, dict) and usage.get("completion_tokens"):
                    usage_tokens = int(usage["completion_tokens"])
                    if usage.get("prompt_tokens") is not None:
                        result.prompt_tokens = int(usage["prompt_tokens"])
    except httpx.HTTPError as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        return result

    end = time.perf_counter()
    if not token_times:
        result.error = "no tokens received"
        return result
    result.ok = True
    result.e2e_s = end - start
    result.ttft_s = token_times[0] - start
    result.itl_s = [b - a for a, b in itertools.pairwise(token_times)]
    if usage_tokens is not None:
        result.output_tokens, result.token_source = usage_tokens, "usage"
    else:
        result.output_tokens, result.token_source = len(token_times), "chunks"
    return result


async def run_level(
    cfg: LoadConfig, prompts: list[str], concurrency: int, num_requests: int
) -> tuple[list[RequestResult], float]:
    """Run ``num_requests`` with at most ``concurrency`` in flight; return results + wall time."""
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(cfg.timeout_s, connect=30.0)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        t0 = time.perf_counter()

        async def _task(i: int) -> RequestResult:
            async with sem:
                return await stream_one(client, cfg, prompts[i % len(prompts)], concurrency, t0)

        results = await asyncio.gather(*(_task(i) for i in range(num_requests)))
        duration = time.perf_counter() - t0
    return list(results), duration


async def warmup(cfg: LoadConfig, prompts: list[str], n: int) -> list[RequestResult]:
    if n <= 0:
        return []
    results, _ = await run_level(cfg, prompts, 1, n)
    return results
