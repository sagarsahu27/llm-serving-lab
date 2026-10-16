"""Model-name routing: ``<backend>/<alias>``, ``auto/<alias>`` or a bare alias."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BackendConfig, GatewayConfig

AUTO_PREFIX = "auto"


class UnknownModelError(LookupError):
    pass


@dataclass(frozen=True)
class Route:
    backend: BackendConfig
    alias: str
    upstream_model: str

    @property
    def public_name(self) -> str:
        return f"{self.backend.name}/{self.alias}"


def _routes_for_alias(config: GatewayConfig, name: str) -> list[Route]:
    """All routes whose alias or upstream model id equals ``name``, in fallback order."""
    routes = []
    for backend in config.ordered_backends():
        for alias, upstream in backend.models.items():
            if name in (alias, upstream):
                routes.append(Route(backend, alias, upstream))
                break
    return routes


def resolve_routes(
    config: GatewayConfig,
    model: str,
    health: dict[str, bool | None] | None = None,
) -> list[Route]:
    """Return candidate routes for ``model``, primary first.

    * ``vllm/qwen2.5-1.5b`` pins a backend; other backends serving the same alias are
      appended only when ``fallback_on_error`` is enabled.
    * ``auto/qwen2.5-1.5b`` or ``qwen2.5-1.5b`` picks any backend serving the alias,
      preferring healthy backends and then ``fallback_order``.
    """
    health = health or {}
    prefix, sep, rest = model.partition("/")

    if sep and prefix in config.backends:
        backend = config.backends[prefix]
        if rest not in backend.models:
            raise UnknownModelError(model)
        primary = Route(backend, rest, backend.models[rest])
        candidates = [primary]
        if config.fallback_on_error:
            candidates += [
                r for r in _routes_for_alias(config, rest) if r.backend.name != backend.name
            ]
        return candidates

    alias = rest if sep and prefix == AUTO_PREFIX else model
    routes = _routes_for_alias(config, alias)
    if not routes:
        raise UnknownModelError(model)

    # Stable sort: healthy (True) first, unknown (None) next, unhealthy (False) last.
    rank = {True: 0, None: 1, False: 2}
    return sorted(routes, key=lambda r: rank[health.get(r.backend.name)])
