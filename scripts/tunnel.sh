#!/usr/bin/env bash
# Forward the gateway and backend ports from the VM to localhost over SSH.
#   ./scripts/tunnel.sh azureuser@<vm-fqdn> [extra ssh args...]
# Then: curl http://localhost:8000/v1/models
set -euo pipefail
HOST="${1:?usage: $0 user@host [ssh args...]}"
shift
echo "Tunnelling gateway :8000, vLLM :8001, SGLang :8002, TGI :8003, Ollama :11434 (Ctrl+C to stop)"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 8000:127.0.0.1:8000 \
  -L 8001:127.0.0.1:8001 \
  -L 8002:127.0.0.1:8002 \
  -L 8003:127.0.0.1:8003 \
  -L 11434:127.0.0.1:11434 \
  "$@" "${HOST}"
