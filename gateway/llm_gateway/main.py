"""FastAPI application: OpenAI-compatible gateway in front of several LLM serving engines."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from . import __version__
from .config import GatewayConfig, load_config
from .health import HealthChecker
from .logging_setup import configure_logging
from .proxy import forward, openai_error
from .routing import AUTO_PREFIX, UnknownModelError, resolve_routes

log = logging.getLogger("llm_gateway.app")


def create_app(
    config: GatewayConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the app. ``config``/``client`` are injectable for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        cfg = config or load_config()
        http = client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
        )
        checker = HealthChecker(cfg, http)
        task = asyncio.create_task(checker.run_forever()) if cfg.health_interval_s > 0 else None
        app.state.config = cfg
        app.state.client = http
        app.state.health = checker
        log.info(
            "gateway started",
            extra={"ctx": {"backends": list(cfg.backends), "version": __version__}},
        )
        try:
            yield
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if client is None:
                await http.aclose()

    app = FastAPI(title="llm-serving-lab gateway", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        response = await call_next(request)
        response.headers.setdefault("x-request-id", request.state.request_id)
        return response

    @app.get("/health")
    async def health(request: Request, refresh: bool = False) -> dict[str, Any]:
        checker: HealthChecker = request.app.state.health
        if refresh:
            await checker.check_all()
        backends = {name: h.to_dict() for name, h in checker.state.items()}
        healthy = [n for n, h in checker.state.items() if h.healthy]
        return {
            "status": "ok" if healthy else "no_healthy_backends",
            "healthy_backends": healthy,
            "backends": backends,
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        cfg: GatewayConfig = request.app.state.config
        health_state = request.app.state.health.snapshot()
        created = int(time.time())
        data: list[dict[str, Any]] = []
        aliases: dict[str, list[str]] = {}
        for backend in cfg.ordered_backends():
            for alias, upstream in backend.models.items():
                aliases.setdefault(alias, []).append(backend.name)
                data.append(
                    {
                        "id": f"{backend.name}/{alias}",
                        "object": "model",
                        "created": created,
                        "owned_by": backend.name,
                        "upstream_model": upstream,
                        "healthy": health_state.get(backend.name),
                    }
                )
        for alias, owners in aliases.items():
            data.append(
                {
                    "id": f"{AUTO_PREFIX}/{alias}",
                    "object": "model",
                    "created": created,
                    "owned_by": "gateway",
                    "backends": owners,
                    "healthy": any(health_state.get(o) for o in owners),
                }
            )
        return {"object": "list", "data": data}

    async def _proxy(request: Request, endpoint: str) -> Response:
        request_id: str = request.state.request_id
        try:
            body = json.loads(await request.body())
        except ValueError:
            return openai_error(400, "request body must be valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return openai_error(400, "request body must be a JSON object", "invalid_request_error")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return openai_error(400, "'model' is required", "invalid_request_error")

        cfg: GatewayConfig = request.app.state.config
        try:
            routes = resolve_routes(cfg, model, request.app.state.health.snapshot())
        except UnknownModelError:
            return openai_error(
                404,
                f"model {model!r} not found; GET /v1/models lists available models",
                "invalid_request_error",
                code="model_not_found",
            )
        return await forward(
            client=request.app.state.client,
            config=cfg,
            routes=routes,
            endpoint=endpoint,
            body=body,
            request_id=request_id,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await _proxy(request, "chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await _proxy(request, "completions")

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled error",
            extra={"ctx": {"request_id": getattr(request.state, "request_id", None)}},
        )
        return openai_error(500, "internal gateway error", "internal_error")

    return app


app = create_app()
