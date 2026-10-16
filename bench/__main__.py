"""CLI: ``python -m bench run ...`` and ``python -m bench report ...``."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

from .loadgen import LoadConfig, run_level, warmup
from .metrics import summarize
from .prompts import load_prompts
from .report import write_report


def _int_list(value: str) -> list[int]:
    try:
        items = [int(v) for v in value.split(",") if v.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers: {value}") from exc
    if not items or any(i < 1 for i in items):
        raise argparse.ArgumentTypeError("concurrency values must be >= 1")
    return items


def _kv(value: str) -> tuple[str, str]:
    key, sep, val = value.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {value!r}")
    return key, val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a concurrency sweep against one endpoint/model")
    run.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible base URL (gateway or a backend directly)",
    )
    run.add_argument(
        "--model", required=True, help="model name, e.g. vllm/qwen2.5-1.5b via the gateway"
    )
    run.add_argument("--label", help="series label in reports (default: model)")
    run.add_argument("--concurrency", type=_int_list, default=[1, 2, 4, 8, 16, 32])
    run.add_argument(
        "--num-prompts",
        type=int,
        default=64,
        help="requests per concurrency level (at least 2x concurrency)",
    )
    run.add_argument("--max-tokens", type=int, default=128)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    run.add_argument("--prompts-file", help='.jsonl ({"prompt": ...}) or .txt (one per line)')
    run.add_argument("--warmup", type=int, default=3, help="sequential warmup requests")
    run.add_argument(
        "--ignore-eos", action="store_true", help="force max_tokens outputs (vLLM/SGLang only)"
    )
    run.add_argument(
        "--no-usage",
        action="store_true",
        help="don't send stream_options.include_usage; count chunks instead",
    )
    run.add_argument("--api-key")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument(
        "--meta",
        type=_kv,
        action="append",
        default=[],
        help="extra key=value recorded in the report (repeatable)",
    )
    run.add_argument("--out", required=True, help="output directory for this run")
    run.add_argument(
        "--max-error-rate",
        type=float,
        default=0.5,
        help="stop the sweep when a level exceeds this error rate",
    )

    rep = sub.add_parser("report", help="combine summary.json files into report.md + charts")
    rep.add_argument("paths", nargs="+", help="summary.json files or directories to search")
    rep.add_argument("--out", required=True)
    return parser


async def _sweep(
    args: argparse.Namespace, cfg: LoadConfig, prompts: list[str], label: str
) -> tuple[list[dict], list[dict]] | None:
    """Warm up, then run every concurrency level. Returns (levels, rows) or None on failure."""
    if args.warmup:
        print(f"[{label}] warmup x{args.warmup}", flush=True)
        warm = await warmup(cfg, prompts, args.warmup)
        failed = [r for r in warm if not r.ok]
        if len(failed) == len(warm):
            print(f"[{label}] warmup failed: {failed[0].error}", file=sys.stderr)
            return None

    levels, rows = [], []
    for conc in args.concurrency:
        n = max(args.num_prompts, conc * 2)
        print(f"[{label}] concurrency={conc} requests={n}", flush=True)
        results, duration = await run_level(cfg, prompts, conc, n)
        summary = summarize(results, duration)
        levels.append(summary)
        rows += [r.to_row() for r in results]
        print(
            f"  ok={summary['succeeded']}/{summary['requests']} "
            f"tok/s={summary['output_throughput_tok_s']} "
            f"ttft_p50={summary['ttft_ms']['p50']}ms tpot_p50={summary['tpot_ms']['p50']}ms",
            flush=True,
        )
        if summary["error_rate"] is not None and summary["error_rate"] > args.max_error_rate:
            print(f"[{label}] error rate {summary['error_rate']:.0%}; stopping sweep")
            break
    return levels, rows


def run_command(args: argparse.Namespace) -> int:
    cfg = LoadConfig(
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        endpoint=args.endpoint,
        include_usage=not args.no_usage,
        ignore_eos=args.ignore_eos,
        api_key=args.api_key,
        timeout_s=args.timeout,
    )
    prompts = load_prompts(args.prompts_file)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or args.model

    outcome = asyncio.run(_sweep(args, cfg, prompts, label))
    if outcome is None:
        return 2
    levels, rows = outcome

    meta = {
        "label": label,
        "model": args.model,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extra": dict(args.meta),
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "levels": levels}, indent=2), encoding="utf-8"
    )
    if rows:
        with (out / "requests.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[{label}] wrote {out / 'summary.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_command(args)
    report = write_report(args.paths, args.out)
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
