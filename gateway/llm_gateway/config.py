"""Gateway configuration: YAML file with ``${VAR:-default}`` environment expansion."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class ConfigError(ValueError):
    """Raised when the gateway configuration is invalid."""


def expand_env(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace ``${VAR}`` / ``${VAR:-default}`` placeholders with environment values."""
    env = os.environ if env is None else env

    def _replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value:
            return value
        if default is None:
            raise ConfigError(f"environment variable {name} is not set and has no default")
        return default

    return _ENV_PATTERN.sub(_replace, text)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str
    models: dict[str, str]
    api_prefix: str = "/v1"
    health_path: str = "/health"
    supports_completions: bool = True
    api_key: str | None = None

    def url(self, endpoint: str) -> str:
        return f"{self.base_url.rstrip('/')}{self.api_prefix}/{endpoint.lstrip('/')}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.health_path}"


@dataclass(frozen=True)
class GatewayConfig:
    backends: dict[str, BackendConfig]
    fallback_order: list[str] = field(default_factory=list)
    fallback_on_error: bool = False
    connect_timeout_s: float = 5.0
    read_timeout_s: float = 300.0
    health_interval_s: float = 10.0
    health_timeout_s: float = 3.0

    def ordered_backends(self) -> list[BackendConfig]:
        """Backends in fallback order, followed by any not listed there."""
        names = [n for n in self.fallback_order if n in self.backends]
        names += [n for n in self.backends if n not in names]
        return [self.backends[n] for n in names]


def _as_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"{key} must be a boolean, got {value!r}")


def parse_config(data: Mapping[str, Any]) -> GatewayConfig:
    raw_backends = data.get("backends") or {}
    if not isinstance(raw_backends, Mapping) or not raw_backends:
        raise ConfigError("config must define at least one backend under 'backends'")

    backends: dict[str, BackendConfig] = {}
    for name, spec in raw_backends.items():
        if "/" in name:
            raise ConfigError(f"backend name {name!r} must not contain '/'")
        if not isinstance(spec, Mapping) or not spec.get("base_url"):
            raise ConfigError(f"backend {name!r} requires a base_url")
        models = spec.get("models") or {}
        if not isinstance(models, Mapping) or not models:
            raise ConfigError(f"backend {name!r} requires a non-empty 'models' mapping")
        backends[name] = BackendConfig(
            name=name,
            base_url=str(spec["base_url"]),
            models={str(k): str(v) for k, v in models.items()},
            api_prefix=str(spec.get("api_prefix", "/v1")),
            health_path=str(spec.get("health_path", "/health")),
            supports_completions=_as_bool(
                spec.get("supports_completions", True), f"{name}.supports_completions"
            ),
            api_key=spec.get("api_key") or None,
        )

    routing = data.get("routing") or {}
    timeouts = data.get("timeouts") or {}
    health = data.get("health") or {}
    order = [str(n) for n in routing.get("fallback_order") or []]
    unknown = [n for n in order if n not in backends]
    if unknown:
        raise ConfigError(f"fallback_order references unknown backends: {unknown}")

    return GatewayConfig(
        backends=backends,
        fallback_order=order,
        fallback_on_error=_as_bool(routing.get("fallback_on_error", False), "fallback_on_error"),
        connect_timeout_s=float(timeouts.get("connect_s", 5.0)),
        read_timeout_s=float(timeouts.get("read_s", 300.0)),
        health_interval_s=float(health.get("interval_s", 10.0)),
        health_timeout_s=float(health.get("timeout_s", 3.0)),
    )


def load_config(path: str | os.PathLike[str] | None = None) -> GatewayConfig:
    config_path = Path(path or os.environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    text = expand_env(config_path.read_text(encoding="utf-8"))
    return parse_config(yaml.safe_load(text) or {})
