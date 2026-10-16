from __future__ import annotations

from pathlib import Path

import pytest

from llm_gateway.config import DEFAULT_CONFIG_PATH, ConfigError, expand_env, load_config
from llm_gateway.routing import UnknownModelError, resolve_routes

from .conftest import make_config


def test_expand_env_uses_value_then_default():
    env = {"A": "x", "EMPTY": ""}
    assert expand_env("${A}-${B:-dflt}-${EMPTY:-e}", env) == "x-dflt-e"
    assert expand_env("${M:-qwen2.5:1.5b}", env) == "qwen2.5:1.5b"
    with pytest.raises(ConfigError):
        expand_env("${MISSING}", env)


def test_default_config_file_loads(monkeypatch):
    monkeypatch.setenv("MODEL_ALIAS", "my-model")
    monkeypatch.setenv("MODEL_ID", "org/My-Model")
    monkeypatch.setenv("GATEWAY_FALLBACK_ON_ERROR", "true")
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert set(cfg.backends) == {"vllm", "sglang", "tgi", "ollama"}
    assert cfg.backends["vllm"].models == {"my-model": "org/My-Model"}
    assert cfg.backends["tgi"].models == {"my-model": "tgi"}
    assert cfg.backends["ollama"].health_url == "http://ollama:11434/api/version"
    assert cfg.backends["sglang"].url("chat/completions") == (
        "http://sglang:30000/v1/chat/completions"
    )
    assert cfg.fallback_on_error is True


def test_invalid_config_rejected(tmp_path: Path):
    bad = tmp_path / "c.yaml"
    bad.write_text("backends:\n  a:\n    base_url: http://x\n")
    with pytest.raises(ConfigError, match="models"):
        load_config(bad)
    bad.write_text(
        "routing:\n  fallback_order: [nope]\n"
        "backends:\n  a:\n    base_url: http://x\n    models: {m: m}\n"
    )
    with pytest.raises(ConfigError, match="unknown backends"):
        load_config(bad)


def test_pinned_route_without_fallback():
    routes = resolve_routes(make_config(), "sglang/qwen")
    assert [r.backend.name for r in routes] == ["sglang"]
    assert routes[0].public_name == "sglang/qwen"


def test_pinned_route_with_fallback_appends_others_in_order():
    routes = resolve_routes(make_config(fallback_on_error=True), "tgi/qwen")
    assert [r.backend.name for r in routes] == ["tgi", "vllm", "sglang", "ollama"]


def test_bare_alias_and_upstream_id_route_by_health():
    cfg = make_config()
    health = {"vllm": False, "sglang": None, "tgi": True, "ollama": True}
    assert [r.backend.name for r in resolve_routes(cfg, "qwen", health)] == [
        "tgi",
        "ollama",
        "sglang",
        "vllm",
    ]
    assert [r.backend.name for r in resolve_routes(cfg, "Qwen/Qwen2.5-1.5B-Instruct")] == [
        "vllm",
        "sglang",
    ]
    assert resolve_routes(cfg, "auto/qwen")[0].backend.name == "vllm"


def test_unknown_models():
    cfg = make_config()
    for name in ("vllm/nope", "nope", "auto/nope"):
        with pytest.raises(UnknownModelError):
            resolve_routes(cfg, name)
