# Concepts: how LLM serving engines work

This lab compares four popular serving stacks behind one OpenAI-compatible gateway. This
page explains what each engine does differently and what the benchmark numbers mean.

## 1. Why LLM serving is hard

Generating text has two very different phases:

| Phase | What happens | Bottleneck |
|---|---|---|
| **Prefill** | The whole prompt is processed in one forward pass, producing the first token and the KV cache for every prompt token. | Compute (matrix multiplies) |
| **Decode** | One token at a time; each step reads all model weights and the KV cache of the sequence. | Memory bandwidth |

At batch size 1, decode is dominated by reading weights from GPU memory: a 1.5B FP16 model is
~3.1 GB, and a T4 has ~320 GB/s, so the best case is roughly 100 tokens/s for one user. The only
way to use the GPU well is to **batch many sequences** so each weight read serves many tokens.
That makes the problem one of KV-cache memory management and scheduling.

### KV cache

For every token in every active sequence, each layer stores a key and a value vector. For
Qwen2.5-1.5B (28 layers, 2 KV heads × 128 dims, GQA) in FP16 that is
`2 (K,V) × 28 × 2 × 128 × 2 bytes ≈ 28 KB per token`. A 4k-token sequence needs ~115 MB, so
the free memory left after loading weights decides how many sequences can run at once.

## 2. Continuous (in-flight) batching

Static batching waits until a whole batch finishes, so short requests sit idle behind long
ones. **Continuous batching**, introduced by Orca and used by vLLM, SGLang and TGI, schedules
at the granularity of a single decode step: finished sequences leave and new ones join on
every iteration. Throughput then grows with concurrency until KV memory or compute runs out.
Many engines also use **chunked prefill**, which splits long prompts into chunks interleaved with
decode steps so that a big prompt does not stall everyone else's token stream (this shows up
as better ITL p99).

## 3. The engines

### vLLM: PagedAttention

* **PagedAttention** stores the KV cache in fixed-size *blocks* (pages), like OS virtual memory.
  A sequence's cache doesn't need contiguous memory, so fragmentation is near zero and memory is
  allocated on demand rather than reserved for `max_model_len`. That allows more concurrent
  sequences.
* Blocks can be shared copy-on-write, which helps parallel sampling, beam search and
  **automatic prefix caching** (hash-identified blocks reused across requests).
* `--gpu-memory-utilization` is the fraction of GPU memory vLLM may use for weights, activations
  and KV blocks; everything left after weights becomes KV cache.
* Also includes CUDA graphs, chunked prefill, speculative decoding, many quantization formats,
  and tensor/pipeline parallelism.

### SGLang: RadixAttention

* **RadixAttention** keeps the KV cache of finished requests in a **radix tree** keyed by token
  sequences, with LRU eviction. Any new request that shares a prefix (system prompt, few-shot
  examples, multi-turn chat history, agent loops) reuses the cached KV instead of recomputing
  it. The scheduler orders requests to maximise prefix hits ("cache-aware scheduling").
* Low-overhead scheduler written to overlap CPU scheduling with GPU work, plus a frontend
  language for structured LLM programs and fast constrained (JSON/regex) decoding.
* `--mem-fraction-static` sets how much memory is pre-allocated for weights + KV pool.
* On a T4 (sm75) we default to `--attention-backend triton`, which is the most portable
  kernel path. On Ampere and newer GPUs, FlashInfer/FA3 backends are faster.

### TGI: Text Generation Inference (Hugging Face)

* Rust router (HTTP, tokenization, continuous batching queue) plus Python model shards. It
  integrates tightly with the HF Hub, with Flash/Paged attention, prefix caching and many quantization
  backends.
* Exposes `/generate`, `/generate_stream` and the OpenAI-compatible **Messages API**
  (`/v1/chat/completions`). It serves one model per container, so the `model` field is ignored.
  We send `"tgi"`.
* Key limits: `--max-input-tokens`, `--max-total-tokens`, `--max-batch-prefill-tokens`;
  `--cuda-memory-fraction` caps memory when sharing a GPU.
* Upstream TGI has moved to maintenance mode, with Hugging Face pointing users to vLLM/SGLang for
  new deployments. It is included here as a widely deployed reference point.

### Ollama: llama.cpp + GGUF

* Wraps **llama.cpp** with a model registry (`ollama pull qwen2.5:1.5b`) and an OpenAI-compatible
  `/v1` API. It is optimised for easy local use on laptops and single GPUs rather than for
  multi-tenant throughput.
* Models are **GGUF** files, usually **quantized** (the default `qwen2.5:1.5b` tag is a 4-bit
  `Q4_K_M` build). Quantization groups weights into blocks with shared scales, which cuts memory
  and bandwidth roughly 3-4× compared with FP16 at a small quality cost.
* Concurrency is bounded by `OLLAMA_NUM_PARALLEL` (parallel slots, each with its own context)
  and `OLLAMA_MAX_QUEUE`.
* ⚠️ **Not apples-to-apples**: Ollama runs a quantized model with a different kernel stack, so
  its numbers mix the effect of quantization with the effect of the engine. Reports flag this.

## 4. Metrics used by the benchmark

| Metric | Definition | What it tells you |
|---|---|---|
| **TTFT** (time to first token) | request sent → first content chunk | Queueing + prefill. This is the "is it responding?" latency. |
| **TPOT** (time per output token) | `(E2E − TTFT) / (output_tokens − 1)` | Decode speed one user sees (1000/TPOT ≈ tokens/s per stream). |
| **ITL** (inter-token latency) | gap between consecutive streamed chunks | Smoothness of streaming; p99 exposes stalls such as prefill interference. |
| **E2E latency** | request sent → stream finished | Depends on output length. |
| **Output throughput** | total output tokens / wall time | Total capacity of the server. |
| **Request throughput** | successful requests / wall time | Capacity in requests (depends on length). |
| **Error rate** | failed / total | Timeouts, 5xx, OOM, queue rejections. |

Percentiles (p50/p90/p99) matter more than means because tail latency is what users notice.

**Closed-loop sweep.** At concurrency *C* the harness keeps *C* requests in flight. As *C* grows,
throughput rises until the engine saturates while TTFT/TPOT rise. The "knee" of that curve is
the useful operating point. Compare engines at equal concurrency and look at the whole curve,
not one number.

**Fairness checklist**

* Same model weights and dtype (FP16 on T4) for vLLM/SGLang/TGI, with the same `max_model_len`.
* One engine on the GPU at a time (`scripts/run_benchmark.sh` enforces this).
* Warmup requests before measuring (CUDA graph capture, JIT, lazy model load in Ollama).
* `temperature=0` and the same prompts. Output lengths still differ unless `--ignore-eos` is
  used (supported by vLLM and SGLang), so prefer per-token metrics.
* Token counts come from the server's `usage` in the final stream chunk when available
  (`stream_options.include_usage`); otherwise streamed chunks are counted.

## 5. The gateway

The FastAPI gateway gives clients one OpenAI-compatible endpoint:

* `model: "vllm/qwen2.5-1.5b"` pins a backend; `auto/qwen2.5-1.5b` (or the bare alias) picks
  the first **healthy** backend in `fallback_order`.
* The gateway rewrites `model` to each backend's upstream id (HF repo for vLLM/SGLang, `tgi`
  for TGI, the GGUF tag for Ollama).
* Streaming responses (SSE) are relayed chunk by chunk without re-encoding, so the gateway adds
  little to TTFT. You can measure the overhead by benchmarking a backend port directly and through
  the gateway.
* Background health checks feed routing and `/health`. If `GATEWAY_FALLBACK_ON_ERROR=true` is
  set, connection errors, timeouts and 5xx responses retry on the next backend before any bytes
  have been streamed.
