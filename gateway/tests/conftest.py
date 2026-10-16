from __future__ import annotations

import pytest
import respx
from fastapi.testclient import TestClient

from llm_gateway.config import parse_config
from llm_gateway.main import create_app

VLLM = "http://vllm.test:8000"
SGLANG = "http://sglang.test:30000"
TGI = "http://tgi.test:80"
OLLAMA = "http://ollama.test:11434"


def make_config(**routing):
    return parse_config(
        {
            "routing": {"fallback_order": ["vllm", "sglang", "tgi", "ollama"], **routing},
            "health": {"interval_s": 0},
            "timeouts": {"connect_s": 1, "read_s": 5},
            "backends": {
                "vllm": {"base_url": VLLM, "models": {"qwen": "Qwen/Qwen2.5-1.5B-Instruct"}},
                "sglang": {"base_url": SGLANG, "models": {"qwen": "Qwen/Qwen2.5-1.5B-Instruct"}},
                "tgi": {"base_url": TGI, "models": {"qwen": "tgi"}},
                "ollama": {
                    "base_url": OLLAMA,
                    "health_path": "/api/version",
                    "supports_completions": False,
                    "models": {"qwen": "qwen2.5:1.5b"},
                },
            },
        }
    )


@pytest.fixture
def mock_backends():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def client_factory(mock_backends):
    clients = []

    def _make(**routing) -> TestClient:
        client = TestClient(create_app(make_config(**routing)))
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(client_factory) -> TestClient:
    return client_factory()


def chat_response(model: str, text: str = "hello") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
