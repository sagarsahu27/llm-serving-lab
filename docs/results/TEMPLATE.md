# Benchmark results: <date> / <GPU>

> Copy this file to `docs/results/<yyyy-mm-dd>-<gpu>.md` and fill it in from
> `results/<run>/report/report.md`. Commit only summaries and charts, not raw CSVs.

## Environment

| Item | Value |
|---|---|
| VM size / GPU | e.g. Standard_NC4as_T4_v3 / 1× T4 16 GB |
| Driver / CUDA | `nvidia-smi` output |
| Model | Qwen/Qwen2.5-1.5B-Instruct (FP16); Ollama: qwen2.5:1.5b (GGUF Q4_K_M) |
| max_model_len | 4096 |
| vLLM image | tag + digest (from report "Extra" column) |
| SGLang image | |
| TGI image | |
| Ollama image | |
| Benchmark | concurrency 1,2,4,8,16,32; 64 prompts/level; max_tokens 128; temperature 0; ignore_eos? |

## Headline numbers

| Engine | Peak out tok/s (@conc) | TTFT p50 @1 | TPOT p50 @1 | TTFT p99 @16 | TPOT p50 @16 | Errors |
|---|---:|---:|---:|---:|---:|---:|
| vLLM | | | | | | |
| SGLang | | | | | | |
| TGI | | | | | | |
| Ollama* | | | | | | |

\* Quantized GGUF, so this is not an apples-to-apples comparison.

## Charts

![Output throughput](../../results/<run>/report/throughput.png)
![TTFT p50](../../results/<run>/report/ttft_p50.png)
![TPOT p50](../../results/<run>/report/tpot_p50.png)

## Observations

- Single-user latency (concurrency 1): which engine decodes fastest, and is that bandwidth-bound?
- Saturation: at what concurrency does each engine's throughput flatten, and how do TTFT and TPOT
  behave past that knee?
- Tail latency: ITL/E2E p99 differences (chunked prefill and scheduler behaviour).
- Memory: KV cache size / max concurrent sequences reported in server logs.
- Gateway overhead: direct-port vs gateway run for one engine.

## Caveats

- Closed-loop load; single GPU; one model size. Results on A100/H100 and larger models can rank
  differently.
- Output lengths differ unless `--ignore-eos` was used.
