"""Aggregate per-request results into benchmark summaries."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestResult:
    concurrency: int
    ok: bool
    status: int | None = None
    error: str | None = None
    start_s: float = 0.0
    ttft_s: float | None = None
    e2e_s: float | None = None
    output_tokens: int = 0
    prompt_tokens: int | None = None
    token_source: str = "chunks"
    itl_s: list[float] = field(default_factory=list)

    @property
    def tpot_s(self) -> float | None:
        """Time per output token after the first one (decode speed seen by one user)."""
        if self.ttft_s is None or self.e2e_s is None or self.output_tokens < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (self.output_tokens - 1)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("itl_s")
        row["tpot_s"] = self.tpot_s
        row["itl_mean_s"] = statistics.fmean(self.itl_s) if self.itl_s else None
        return row


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Linear-interpolated percentile (same definition as numpy's default)."""
    data = sorted(v for v in values if v is not None and not math.isnan(v))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * pct / 100.0
    low, high = math.floor(rank), math.ceil(rank)
    return data[low] + (data[high] - data[low]) * (rank - low)


def _dist(values: Sequence[float], scale: float = 1000.0) -> dict[str, float | None]:
    """mean/p50/p90/p99 of ``values`` (seconds) converted to ms by default."""
    clean = [v for v in values if v is not None]

    def _s(v: float | None) -> float | None:
        return None if v is None else round(v * scale, 3)

    return {
        "mean": _s(statistics.fmean(clean)) if clean else None,
        "p50": _s(percentile(clean, 50)),
        "p90": _s(percentile(clean, 90)),
        "p99": _s(percentile(clean, 99)),
    }


def summarize(results: Sequence[RequestResult], duration_s: float) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    total_out = sum(r.output_tokens for r in ok)
    itls = [x for r in ok for x in r.itl_s]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            key = (r.error or f"HTTP {r.status}")[:120]
            errors[key] = errors.get(key, 0) + 1
    return {
        "concurrency": results[0].concurrency if results else None,
        "requests": len(results),
        "succeeded": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else None,
        "duration_s": round(duration_s, 3),
        "request_throughput_rps": round(len(ok) / duration_s, 3) if duration_s > 0 else None,
        "output_throughput_tok_s": round(total_out / duration_s, 2) if duration_s > 0 else None,
        "total_output_tokens": total_out,
        "mean_output_tokens": round(total_out / len(ok), 2) if ok else None,
        "token_source": sorted({r.token_source for r in ok}),
        "ttft_ms": _dist([r.ttft_s for r in ok]),
        "tpot_ms": _dist([r.tpot_s for r in ok]),
        "itl_ms": _dist(itls),
        "e2e_ms": _dist([r.e2e_s for r in ok]),
        "errors": errors,
    }
