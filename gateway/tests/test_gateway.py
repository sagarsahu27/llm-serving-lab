from __future__ import annotations

import json

import httpx

from .conftest import OLLAMA, SGLANG, TGI, VLLM, chat_response


def _chat(model: str, **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_chat_completion_routes_to_pinned_backend(client, mock_backends):
    route = mock_backends.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("Qwen/Qwen2.5-1.5B-Instruct"))
    )
    resp = client.post("/v1/chat/completions", json=_chat("vllm/qwen", temperature=0))

    assert resp.status_code == 200
    assert resp.headers["x-gateway-backend"] == "vllm"
    assert resp.json()["model"] == "vllm/qwen"
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert sent["temperature"] == 0


def test_tgi_and_ollama_receive_their_upstream_model_ids(client, mock_backends):
    tgi = mock_backends.post(f"{TGI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("tgi"))
    )
    ollama = mock_backends.post(f"{OLLAMA}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("qwen2.5:1.5b"))
    )
    assert client.post("/v1/chat/completions", json=_chat("tgi/qwen")).status_code == 200
    assert client.post("/v1/chat/completions", json=_chat("ollama/qwen")).status_code == 200
    assert json.loads(tgi.calls.last.request.content)["model"] == "tgi"
    assert json.loads(ollama.calls.last.request.content)["model"] == "qwen2.5:1.5b"


def test_streaming_is_passed_through_unchanged(client, mock_backends):
    chunks = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    mock_backends.post(f"{SGLANG}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"".join(chunks)),
        )
    )
    with client.stream(
        "POST", "/v1/chat/completions", json=_chat("sglang/qwen", stream=True)
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-gateway-backend"] == "sglang"
        body = b"".join(resp.iter_bytes())
    assert body == b"".join(chunks)


def test_completions_endpoint(client, mock_backends):
    mock_backends.post(f"{VLLM}/v1/completions").mock(
        return_value=httpx.Response(
            200, json={"object": "text_completion", "model": "x", "choices": [{"text": "ok"}]}
        )
    )
    resp = client.post("/v1/completions", json={"model": "vllm/qwen", "prompt": "hi"})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["text"] == "ok"


def test_completions_unsupported_backend_returns_400(client):
    resp = client.post("/v1/completions", json={"model": "ollama/qwen", "prompt": "hi"})
    assert resp.status_code == 400
    assert "does not support" in resp.json()["error"]["message"]


def test_unknown_model_returns_openai_style_404(client):
    resp = client.post("/v1/chat/completions", json=_chat("vllm/does-not-exist"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_invalid_requests_return_400(client):
    assert client.post("/v1/chat/completions", content=b"{not json").status_code == 400
    assert client.post("/v1/chat/completions", json=[1, 2]).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_upstream_4xx_is_passed_through_without_fallback(client_factory, mock_backends):
    client = client_factory(fallback_on_error=True)
    mock_backends.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad max_tokens"}})
    )
    sglang = mock_backends.post(f"{SGLANG}/v1/chat/completions")
    resp = client.post("/v1/chat/completions", json=_chat("vllm/qwen"))
    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "bad max_tokens"
    assert not sglang.called


def test_connect_error_without_fallback_returns_502(client, mock_backends):
    mock_backends.post(f"{VLLM}/v1/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    resp = client.post("/v1/chat/completions", json=_chat("vllm/qwen"))
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "backend_unavailable"


def test_timeout_returns_504(client, mock_backends):
    mock_backends.post(f"{VLLM}/v1/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    assert client.post("/v1/chat/completions", json=_chat("vllm/qwen")).status_code == 504


def test_fallback_on_connect_error_and_5xx(client_factory, mock_backends):
    client = client_factory(fallback_on_error=True)
    mock_backends.post(f"{VLLM}/v1/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    mock_backends.post(f"{SGLANG}/v1/chat/completions").mock(
        return_value=httpx.Response(503, json={"error": "loading"})
    )
    mock_backends.post(f"{TGI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("tgi"))
    )
    resp = client.post("/v1/chat/completions", json=_chat("vllm/qwen"))
    assert resp.status_code == 200
    assert resp.headers["x-gateway-backend"] == "tgi"
    assert resp.json()["model"] == "tgi/qwen"
    assert "vllm" in resp.headers["x-gateway-fallback-from"]
    assert "sglang: HTTP 503" in resp.headers["x-gateway-fallback-from"]


def test_last_backend_5xx_is_returned_as_is(client, mock_backends):
    mock_backends.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "oom"}})
    )
    resp = client.post("/v1/chat/completions", json=_chat("vllm/qwen"))
    assert resp.status_code == 500
    assert resp.json()["error"]["message"] == "oom"


def test_auto_alias_prefers_healthy_backend(client, mock_backends):
    mock_backends.get(f"{VLLM}/health").mock(side_effect=httpx.ConnectError("down"))
    mock_backends.get(f"{SGLANG}/health").mock(side_effect=httpx.ConnectError("down"))
    mock_backends.get(f"{TGI}/health").mock(return_value=httpx.Response(503))
    mock_backends.get(f"{OLLAMA}/api/version").mock(
        return_value=httpx.Response(200, json={"version": "x"})
    )
    ollama = mock_backends.post(f"{OLLAMA}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("qwen2.5:1.5b"))
    )

    health = client.get("/health", params={"refresh": "true"}).json()
    assert health["healthy_backends"] == ["ollama"]
    assert health["backends"]["tgi"]["error"] == "HTTP 503"

    resp = client.post("/v1/chat/completions", json=_chat("auto/qwen"))
    assert resp.status_code == 200
    assert resp.headers["x-gateway-backend"] == "ollama"
    assert ollama.called


def test_models_lists_backend_and_auto_entries(client):
    data = client.get("/v1/models").json()
    ids = {m["id"] for m in data["data"]}
    assert {"vllm/qwen", "sglang/qwen", "tgi/qwen", "ollama/qwen", "auto/qwen"} <= ids
    vllm = next(m for m in data["data"] if m["id"] == "vllm/qwen")
    assert vllm["upstream_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_request_id_is_propagated(client, mock_backends):
    route = mock_backends.post(f"{VLLM}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("x"))
    )
    resp = client.post(
        "/v1/chat/completions", json=_chat("vllm/qwen"), headers={"x-request-id": "abc123"}
    )
    assert resp.headers["x-request-id"] == "abc123"
    assert route.calls.last.request.headers["x-request-id"] == "abc123"
