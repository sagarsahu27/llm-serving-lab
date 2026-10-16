"""Combine one or more benchmark runs into a markdown report with charts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

CHARTS = [
    ("output_throughput_tok_s", None, "Output throughput (tok/s)", "throughput.png"),
    ("ttft_ms", "p50", "TTFT p50 (ms)", "ttft_p50.png"),
    ("tpot_ms", "p50", "TPOT p50 (ms/token)", "tpot_p50.png"),
    ("e2e_ms", "p90", "E2E latency p90 (ms)", "e2e_p90.png"),
    ("request_throughput_rps", None, "Request throughput (req/s)", "rps.png"),
]


def find_runs(paths: list[str]) -> list[dict[str, Any]]:
    runs = []
    for p in paths:
        path = Path(p)
        files = [path] if path.is_file() else sorted(path.rglob("summary.json"))
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_path"] = str(f)
            runs.append(data)
    if not runs:
        raise FileNotFoundError(f"no summary.json found under {paths}")
    return runs


def _metric(level: dict[str, Any], key: str, sub: str | None) -> float | None:
    value = level.get(key)
    if sub is not None:
        value = (value or {}).get(sub)
    return value


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def render_charts(runs: list[dict[str, Any]], out_dir: Path) -> list[tuple[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts")
        return []

    written = []
    for key, sub, title, filename in CHARTS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for run in runs:
            levels = sorted(run["levels"], key=lambda lv: lv["concurrency"])
            xs = [lv["concurrency"] for lv in levels]
            ys = [_metric(lv, key, sub) for lv in levels]
            ax.plot(xs, ys, marker="o", label=run["meta"]["label"])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency (in-flight requests)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        written.append((title, filename))
    return written


def build_markdown(runs: list[dict[str, Any]], charts: list[tuple[str, str]]) -> str:
    lines = [
        "# LLM serving benchmark report",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_",
        "",
        "## Setup",
        "",
        "| Label | Model | Endpoint | max_tokens | ignore_eos | Prompts | Extra |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        m = run["meta"]
        extra = ", ".join(f"{k}={v}" for k, v in (m.get("extra") or {}).items()) or "-"
        lines.append(
            f"| {m['label']} | `{m['model']}` | {m['base_url']} | {m['max_tokens']} | "
            f"{m.get('ignore_eos', False)} | {m.get('num_prompts', '-')} | {extra} |"
        )

    lines += [
        "",
        "## Results",
        "",
        "Latencies in ms. TTFT = time to first token, TPOT = time per output token after the "
        "first, E2E = full request latency.",
        "",
        "| Label | Conc. | OK/Total | Err % | req/s | out tok/s | TTFT p50 | TTFT p99 | "
        "TPOT p50 | TPOT p99 | ITL p99 | E2E p50 | E2E p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        label = run["meta"]["label"]
        for lv in sorted(run["levels"], key=lambda x: x["concurrency"]):
            err = lv.get("error_rate")
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        str(lv["concurrency"]),
                        f"{lv['succeeded']}/{lv['requests']}",
                        _fmt(err * 100 if err is not None else None),
                        _fmt(lv["request_throughput_rps"], 2),
                        _fmt(lv["output_throughput_tok_s"]),
                        _fmt(lv["ttft_ms"]["p50"]),
                        _fmt(lv["ttft_ms"]["p99"]),
                        _fmt(lv["tpot_ms"]["p50"], 2),
                        _fmt(lv["tpot_ms"]["p99"], 2),
                        _fmt(lv["itl_ms"]["p99"], 2),
                        _fmt(lv["e2e_ms"]["p50"]),
                        _fmt(lv["e2e_ms"]["p99"]),
                    ]
                )
                + " |"
            )

    lines += ["", "## Peak throughput", "", "| Label | Peak out tok/s | At concurrency |"]
    lines.append("|---|---:|---:|")
    for run in runs:
        levels = [lv for lv in run["levels"] if lv.get("output_throughput_tok_s")]
        if levels:
            best = max(levels, key=lambda lv: lv["output_throughput_tok_s"])
            lines.append(
                f"| {run['meta']['label']} | {_fmt(best['output_throughput_tok_s'])} | "
                f"{best['concurrency']} |"
            )

    if charts:
        lines += ["", "## Charts", ""]
        for title, filename in charts:
            lines += [f"### {title}", "", f"![{title}]({filename})", ""]

    caveats = [
        "Each backend was benchmarked alone on the GPU unless the labels say otherwise.",
        "Closed-loop load: each concurrency level keeps N requests in flight; these are not "
        "open-loop arrival-rate (Poisson) results.",
    ]
    if any("ollama" in r["meta"]["label"].lower() for r in runs):
        caveats.append(
            "**Ollama is not apples-to-apples**: it serves a quantized GGUF build (llama.cpp) "
            "of the model, while vLLM/SGLang/TGI serve the original FP16 weights. Compare "
            "its numbers as 'developer-friendly local runtime' rather than as an equal engine."
        )
    sources = {s for r in runs for lv in r["levels"] for s in lv.get("token_source", [])}
    if "chunks" in sources:
        caveats.append(
            "Some runs counted output tokens by streamed chunks (no `usage` in the stream), "
            "which can undercount if a server packs several tokens per chunk."
        )
    if not all(r["meta"].get("ignore_eos") for r in runs):
        caveats.append(
            "Without `--ignore-eos`, output lengths differ between engines; prefer per-token "
            "metrics (TPOT, tok/s) over E2E when comparing."
        )
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in caveats] + [""]
    return "\n".join(lines)


def write_report(paths: list[str], out_dir: str) -> Path:
    runs = find_runs(paths)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    charts = render_charts(runs, out)
    report = out / "report.md"
    report.write_text(build_markdown(runs, charts), encoding="utf-8")
    return report
