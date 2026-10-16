"""Forward OpenAI-compatible requests to a backend, with optional fallback and SSE passthrough."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import GatewayConfig
from .routing import Route

log = logging.getLogger("llm_gateway.proxy")

_PASSTHROUGH_HEADERS = ("content-type", "cache-control")


def openai_error(status: int, message: str, err_type: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": err_type, "param": None, "code": code}},
    )


def _gateway_headers(route: Route, request_id: str) -> dict[str, str]:
    return {
        "x-request-id": request_id,
        "x-gateway-backend": route.backend.name,
        "x-gateway-model": route.public_name,
    }


def _rewrite_model(content: bytes, public_name: str) -> bytes:
    try:
        data = json.loads(content)
    except ValueError:
        return content
    if isinstance(data, dict) and "model" in data:
        data["model"] = public_name
        return json.dumps(data).encode()
    return content


async def _relay_stream(
    resp: httpx.Response, ctx: dict[str, Any], start: float
) -> AsyncIterator[bytes]:
    first_byte: float | None = None
    outcome = "ok"
    try:
        async for chunk in resp.aiter_raw():
            if first_byte is None:
                first_byte = time.perf_counter()
            yield chunk
    except httpx.HTTPError as exc:
        outcome = "upstream_stream_error"
        log.warning("upstream stream failed", extra={"ctx": {**ctx, "error": str(exc)}})
        err = {"error": {"message": f"upstream stream error: {exc}", "type": "upstream_error"}}
        yield f"data: {json.dumps(err)}\n\n".encode()
    except BaseException:
        outcome = "client_disconnected"
        raise
    finally:
        await resp.aclose()
        end = time.perf_counter()
        log.info(
            "request completed",
            extra={
                "ctx": {
                    **ctx,
                    "status": resp.status_code,
                    "outcome": outcome,
                    "ttfb_ms": round((first_byte - start) * 1000, 2) if first_byte else None,
                    "latency_ms": round((end - start) * 1000, 2),
                }
            },
        )


async def forward(
    *,
    client: httpx.AsyncClient,
    config: GatewayConfig,
    routes: list[Route],
    endpoint: str,
    body: dict[str, Any],
    request_id: str,
) -> Response:
    """Send ``body`` to the first route that answers; fall back on transport errors / 5xx."""
    stream = bool(body.get("stream"))
    timeout = httpx.Timeout(
        connect=config.connect_timeout_s,
        read=config.read_timeout_s,
        write=config.read_timeout_s,
        pool=config.connect_timeout_s,
    )
    attempts: list[str] = []
    last_failure: tuple[int, str] | None = None

    for index, route in enumerate(routes):
        is_last = index == len(routes) - 1
        backend = route.backend
        if endpoint == "completions" and not backend.supports_completions:
            attempts.append(f"{backend.name}: /v1/completions not supported")
            last_failure = (400, f"backend {backend.name} does not support /v1/completions")
            continue

        payload = {**body, "model": route.upstream_model}
        headers = {"content-type": "application/json", "x-request-id": request_id}
        if backend.api_key:
            headers["authorization"] = f"Bearer {backend.api_key}"
        ctx = {
            "request_id": request_id,
            "backend": backend.name,
            "model": route.public_name,
            "endpoint": endpoint,
            "stream": stream,
            "attempt": index + 1,
        }

        start = time.perf_counter()
        request = client.build_request(
            "POST", backend.url(endpoint), json=payload, headers=headers, timeout=timeout
        )
        try:
            resp = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            attempts.append(f"{backend.name}: timeout")
            last_failure = (504, f"backend {backend.name} timed out: {exc}")
            log.warning("upstream timeout", extra={"ctx": {**ctx, "error": str(exc)}})
            continue
        except httpx.TransportError as exc:
            attempts.append(f"{backend.name}: {type(exc).__name__}")
            last_failure = (502, f"backend {backend.name} unreachable: {exc}")
            log.warning("upstream unreachable", extra={"ctx": {**ctx, "error": str(exc)}})
            continue

        if resp.status_code >= 500 and not is_last:
            await resp.aread()
            await resp.aclose()
            attempts.append(f"{backend.name}: HTTP {resp.status_code}")
            log.warning(
                "upstream 5xx, falling back",
                extra={"ctx": {**ctx, "status": resp.status_code}},
            )
            continue

        out_headers = _gateway_headers(route, request_id)
        if attempts:
            out_headers["x-gateway-fallback-from"] = "; ".join(attempts)

        if stream and resp.is_success:
            for name in _PASSTHROUGH_HEADERS:
                if name in resp.headers:
                    out_headers[name] = resp.headers[name]
            return StreamingResponse(
                _relay_stream(resp, ctx, start),
                status_code=resp.status_code,
                headers=out_headers,
                media_type=resp.headers.get("content-type", "text/event-stream"),
            )

        content = await resp.aread()
        await resp.aclose()
        if resp.is_success:
            content = _rewrite_model(content, route.public_name)
        log.info(
            "request completed",
            extra={
                "ctx": {
                    **ctx,
                    "status": resp.status_code,
                    "outcome": "ok" if resp.is_success else "upstream_error",
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                }
            },
        )
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=out_headers,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    status, message = last_failure or (503, "no backend available")
    if len(attempts) > 1:
        message = f"all backends failed ({'; '.join(attempts)})"
    response = openai_error(
        status,
        message,
        "invalid_request_error" if status == 400 else "backend_unavailable",
    )
    response.headers["x-request-id"] = request_id
    log.warning(
        "request failed",
        extra={"ctx": {"request_id": request_id, "status": status, "attempts": attempts}},
    )
    return response
