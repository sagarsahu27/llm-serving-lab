from __future__ import annotations

import json

import httpx
import pytest
import respx

from bench.__main__ import main
from bench.loadgen import LoadConfig, run_level, stream_one
from bench.metrics import RequestResult, percentile, summarize
from bench.report import build_markdown, find_runs

BASE = "http://bench.test/v1"


def sse(*events: dict | str) -> bytes:
    out = b""
    for e in events:
        data = e if isinstance(e, str) else json.dumps(e)
        out += f"data: {data}\n\n".encode()
    return out


def delta(text: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def test_percentile_matches_linear_interpolation():
    assert percentile([], 50) is None
    assert percentile([5], 99) == 5
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4, 5], 90) == pytest.approx(4.6)


def test_tpot_and_summary():
    r1 = RequestResult(concurrency=4, ok=True, ttft_s=0.1, e2e_s=1.1, output_tokens=11)
    r1.itl_s = [0.1] * 10
    r2 = RequestResult(concurrency=4, ok=False, error="boom")
    assert r1.tpot_s == pytest.approx(0.1)
    s = summarize([r1, r2], duration_s=2.0)
    assert s["concurrency"] == 4
    assert s["succeeded"] == 1 and s["error_rate"] == 0.5
    assert s["output_throughput_tok_s"] == 5.5
    assert s["ttft_ms"]["p50"] == 100.0
    assert s["tpot_ms"]["p50"] == 100.0
    assert s["errors"] == {"boom": 1}
    assert "itl_mean_s" in r1.to_row()


def test_payload_options():
    cfg = LoadConfig(base_url=BASE, model="m", ignore_eos=True, extra_body={"top_p": 0.9})
    body = cfg.payload("hi")
    assert cfg.url == f"{BASE}/chat/completions"
    assert body["messages"][0]["content"] == "hi"
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["ignore_eos"] is True and body["top_p"] == 0.9
    comp = LoadConfig(base_url=BASE, model="m", endpoint="completions", include_usage=False)
    assert comp.url == f"{BASE}/completions"
    assert comp.payload("x")["prompt"] == "x" and "stream_options" not in comp.payload("x")


@respx.mock
async def test_stream_one_uses_usage_when_present():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                {"choices": [{"delta": {"role": "assistant"}}]},
                delta("Hel"),
                delta("lo"),
                delta("!"),
                {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
                "[DONE]",
            ),
        )
    )
    async with httpx.AsyncClient() as client:
        r = await stream_one(client, LoadConfig(base_url=BASE, model="m"), "hi", 1, 0.0)
    assert r.ok, r.error
    assert r.output_tokens == 3 and r.token_source == "usage" and r.prompt_tokens == 7
    assert len(r.itl_s) == 2
    assert 0 <= r.ttft_s <= r.e2e_s


@respx.mock
async def test_stream_one_counts_chunks_and_reports_errors():
    route = respx.post(f"{BASE}/chat/completions")
    route.mock(return_value=httpx.Response(200, content=sse(delta("a"), delta("b"), "[DONE]")))
    cfg = LoadConfig(base_url=BASE, model="m", include_usage=False)
    async with httpx.AsyncClient() as client:
        ok = await stream_one(client, cfg, "hi", 1, 0.0)
        assert ok.ok and ok.output_tokens == 2 and ok.token_source == "chunks"

        route.mock(return_value=httpx.Response(429, text="slow down"))
        err = await stream_one(client, cfg, "hi", 1, 0.0)
        assert not err.ok and err.status == 429 and "slow down" in err.error

        route.mock(return_value=httpx.Response(200, content=sse({"error": {"message": "x"}})))
        err = await stream_one(client, cfg, "hi", 1, 0.0)
        assert not err.ok and "x" in err.error

        route.mock(side_effect=httpx.ConnectError("refused"))
        err = await stream_one(client, cfg, "hi", 1, 0.0)
        assert not err.ok and "ConnectError" in err.error


@respx.mock
async def test_run_level_runs_all_requests():
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse(delta("a"), delta("b"), "[DONE]"))
    )
    results, duration = await run_level(LoadConfig(base_url=BASE, model="m"), ["p"], 4, 10)
    assert len(results) == 10 and all(r.ok for r in results)
    assert duration > 0


def test_cli_run_and_report_end_to_end(tmp_path):
    with respx.mock:
        respx.post(f"{BASE}/chat/completions").mock(
            return_value=httpx.Response(200, content=sse(delta("a"), delta("b"), "[DONE]"))
        )
        for label in ("vllm", "ollama"):
            rc = main(
                [
                    "run",
                    "--base-url",
                    BASE,
                    "--model",
                    f"{label}/m",
                    "--label",
                    label,
                    "--concurrency",
                    "1,2",
                    "--num-prompts",
                    "4",
                    "--warmup",
                    "1",
                    "--meta",
                    "gpu=T4",
                    "--out",
                    str(tmp_path / "runs" / label),
                ]
            )
            assert rc == 0
    summary = json.loads((tmp_path / "runs" / "vllm" / "summary.json").read_text())
    assert [lv["concurrency"] for lv in summary["levels"]] == [1, 2]
    assert summary["meta"]["extra"] == {"gpu": "T4"}
    assert (tmp_path / "runs" / "vllm" / "requests.csv").exists()

    assert main(["report", str(tmp_path / "runs"), "--out", str(tmp_path / "report")]) == 0
    report = (tmp_path / "report" / "report.md").read_text()
    assert "| vllm | 1 |" in report
    assert "Ollama is not apples-to-apples" in report
    assert (tmp_path / "report" / "throughput.png").exists()


def test_markdown_without_charts(tmp_path):
    summary = {
        "meta": {
            "label": "sglang",
            "model": "m",
            "base_url": BASE,
            "max_tokens": 8,
            "ignore_eos": True,
        },
        "levels": [
            summarize(
                [
                    RequestResult(
                        concurrency=1,
                        ok=True,
                        ttft_s=0.01,
                        e2e_s=0.1,
                        output_tokens=8,
                        token_source="usage",
                    )
                ],
                0.1,
            )
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    md = build_markdown(find_runs([str(path)]), charts=[])
    assert "## Charts" not in md
    assert "| sglang | 80.0 | 1 |" in md
    assert "apples-to-apples" not in md
