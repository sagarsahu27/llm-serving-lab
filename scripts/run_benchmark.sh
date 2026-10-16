#!/usr/bin/env bash
# Benchmark each backend ALONE on the GPU (fair comparison), then build a combined report.
#
#   ./scripts/run_benchmark.sh
#   BACKENDS="vllm sglang" CONCURRENCY=1,8,32 MAX_TOKENS=256 ./scripts/run_benchmark.sh
#
# Env: BACKENDS, CONCURRENCY, NUM_PROMPTS, MAX_TOKENS, RUN_DIR, PY, BENCH_ARGS (extra flags
# for "python -m bench run", e.g. "--ignore-eos"), WAIT_TIMEOUT (seconds per backend start).
set -euo pipefail
cd "$(dirname "$0")/.."

BACKENDS="${BACKENDS:-vllm sglang tgi ollama}"
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
MAX_TOKENS="${MAX_TOKENS:-128}"
RUN_DIR="${RUN_DIR:-results/$(date -u +%Y%m%d-%H%M%S)}"
PY="${PY:-.venv/bin/python}"
BENCH_ARGS="${BENCH_ARGS:-}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
GATEWAY="${GATEWAY_URL:-http://127.0.0.1:8000}"
ENV_FILE="compose/.env"
COMPOSE=(docker compose -f compose/docker-compose.yml --env-file "${ENV_FILE}")

[[ -f "${ENV_FILE}" ]] || { echo "missing ${ENV_FILE}: cp compose/.env.example ${ENV_FILE}" >&2; exit 1; }
[[ -x "${PY}" ]] || { echo "missing ${PY}: run 'make venv'" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

env_value() { grep -E "^$1=" "${ENV_FILE}" | tail -n1 | cut -d= -f2- || true; }
ALIAS="$(env_value MODEL_ALIAS)"; ALIAS="${ALIAS:-qwen2.5-1.5b}"
GPU="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo unknown)"
mkdir -p "${RUN_DIR}"
echo "Results -> ${RUN_DIR} | GPU: ${GPU}"

stop_backends() {
  "${COMPOSE[@]}" --profile all stop vllm sglang tgi ollama >/dev/null 2>&1 || true
}
trap stop_backends EXIT

"${COMPOSE[@]}" up -d --build --wait gateway

for backend in ${BACKENDS}; do
  echo "================ ${backend} ================"
  stop_backends
  "${COMPOSE[@]}" --profile "${backend}" up -d --wait --wait-timeout "${WAIT_TIMEOUT}" "${backend}"
  if [[ "${backend}" == "ollama" ]]; then
    "${COMPOSE[@]}" --profile ollama run --rm ollama-pull
  fi

  for _ in $(seq 1 60); do
    if curl -fsS "${GATEWAY}/health?refresh=true" \
      | jq -e --arg b "${backend}" '.healthy_backends | index($b)' >/dev/null; then
      break
    fi
    sleep 5
  done

  cid="$("${COMPOSE[@]}" ps -q "${backend}")"
  image="$(docker inspect -f '{{.Config.Image}}' "${cid}")"
  digest="$(docker image inspect -f '{{join .RepoDigests ","}}' "$(docker inspect -f '{{.Image}}' "${cid}")")"
  docker logs "${cid}" >"${RUN_DIR}/${backend}.server.log" 2>&1 || true

  # shellcheck disable=SC2086  # BENCH_ARGS is intentionally word-split
  "${PY}" -m bench run \
    --base-url "${GATEWAY}/v1" \
    --model "${backend}/${ALIAS}" \
    --label "${backend}" \
    --concurrency "${CONCURRENCY}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-tokens "${MAX_TOKENS}" \
    --meta "gpu=${GPU}" --meta "image=${image}" --meta "digest=${digest:-n/a}" \
    --out "${RUN_DIR}/${backend}" \
    ${BENCH_ARGS} || echo "benchmark for ${backend} failed; continuing"

  nvidia-smi --query-gpu=memory.used --format=csv,noheader >"${RUN_DIR}/${backend}.gpu-mem.txt" 2>/dev/null || true
done

"${PY}" -m bench report "${RUN_DIR}" --out "${RUN_DIR}/report"
echo "Report: ${RUN_DIR}/report/report.md"
