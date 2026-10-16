#!/usr/bin/env bash
# Smoke-test the gateway: health, model list, and a chat (plain + streaming) per healthy backend.
#   ./scripts/smoke_test.sh [gateway_url]        (default http://127.0.0.1:8000)
set -euo pipefail
GW="${1:-${GATEWAY_URL:-http://127.0.0.1:8000}}"
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

echo "== GET ${GW}/health?refresh=true"
health="$(curl -fsS "${GW}/health?refresh=true")"
echo "${health}" | jq '{status, healthy_backends}'

echo "== GET ${GW}/v1/models"
curl -fsS "${GW}/v1/models" | jq -r '.data[] | "\(.id)\thealthy=\(.healthy)"'

mapfile -t backends < <(echo "${health}" | jq -r '.healthy_backends[]')
if [[ ${#backends[@]} -eq 0 ]]; then
  echo "no healthy backends - start one with: make up-vllm" >&2
  exit 1
fi

failures=0
for backend in "${backends[@]}"; do
  model="$(curl -fsS "${GW}/v1/models" | jq -r --arg b "${backend}" '[.data[] | select(.owned_by == $b)][0].id')"
  echo "== ${model}: chat completion"
  body="$(jq -nc --arg m "${model}" \
    '{model: $m, max_tokens: 32, temperature: 0, messages: [{role: "user", content: "Say hello in five words."}]}')"
  if out="$(curl -fsS "${GW}/v1/chat/completions" -H 'content-type: application/json' -d "${body}")"; then
    echo "${out}" | jq -r '"  reply: \(.choices[0].message.content | gsub("\n"; " "))\n  usage: \(.usage)"'
  else
    echo "  FAILED"; failures=$((failures + 1)); continue
  fi

  echo "== ${model}: streaming"
  stream_body="$(echo "${body}" | jq -c '. + {stream: true}')"
  chunks="$(curl -fsSN "${GW}/v1/chat/completions" -H 'content-type: application/json' \
    -d "${stream_body}" | grep -c '^data:' || true)"
  if [[ "${chunks}" -gt 1 ]]; then
    echo "  received ${chunks} SSE events"
  else
    echo "  FAILED (${chunks} events)"; failures=$((failures + 1))
  fi
done

if [[ ${failures} -gt 0 ]]; then
  echo "${failures} check(s) failed"
  exit 1
fi
echo "smoke test passed"
