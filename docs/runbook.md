# Runbook: provision → deploy → smoke test → benchmark → teardown

Everything GPU-related runs on an Azure GPU VM. Your local machine only needs Azure CLI,
an SSH client and (optionally) this repo for editing.

## 0. Prerequisites (local machine)

* Azure CLI logged in: `az login`, then `az account set -s <subscription>`.
* An SSH key: `ssh-keygen -t ed25519` (creates `~/.ssh/id_ed25519.pub`).
* GPU quota in your region. New subscriptions usually have **0** GPU quota:
  ```bash
  LOCATION=eastus ./infra/deploy.sh quota
  ```
  `Standard_NC4as_T4_v3` needs 4 vCPUs of **Standard NCASv3_T4 Family**;
  `Standard_NC24ads_A100_v4` needs 24 vCPUs of **Standard NCADS_A100_v4 Family**. Request an
  increase in the Azure portal → *Quotas* → *Compute* if needed.
* On Windows, run the bash scripts from WSL, Git Bash or Azure Cloud Shell, or use the raw
  `az` commands below in PowerShell.

## 1. Provision the VM

```bash
# defaults: rg-llm-serving-lab, eastus, Standard_NC4as_T4_v3, auto-shutdown 19:00 UTC
./infra/deploy.sh up
# A100 instead:  VM_SIZE=Standard_NC24ads_A100_v4 ./infra/deploy.sh up
# Spot pricing:  USE_SPOT=true ./infra/deploy.sh up
```

<details><summary>Equivalent raw az commands (PowerShell-friendly)</summary>

```powershell
az group create -n rg-llm-serving-lab -l eastus
$myIp = (Invoke-RestMethod https://api.ipify.org)
az deployment group create -g rg-llm-serving-lab -n llm-serving-lab `
  --template-file infra/main.bicep `
  --parameters sshPublicKey="$(Get-Content $HOME\.ssh\id_ed25519.pub)" `
               allowedSshSourcePrefix="$myIp/32" `
               repoUrl=https://github.com/sagarsahu27/llm-serving-lab.git
az deployment group show -g rg-llm-serving-lab -n llm-serving-lab --query properties.outputs
```
</details>

What gets created: VNet + NSG (**inbound SSH only, from your IP**), static public IP with DNS
name, NIC, Ubuntu 22.04 Gen2 VM with a 256 GB Premium SSD, and a daily auto-shutdown schedule.

On first boot, cloud-init runs `scripts/bootstrap_vm.sh`. It installs the NVIDIA datacenter
driver (Ubuntu-signed, default branch `580-server`), Docker Engine with the compose plugin, and
the NVIDIA Container Toolkit, clones the repo, and **reboots once**. This takes about 10 minutes.

```bash
ssh azureuser@<fqdn>
cloud-init status --wait            # wait for "status: done" (connection drops at reboot)
nvidia-smi                          # GPU visible?
docker run --rm --gpus all ubuntu nvidia-smi   # GPU visible inside containers?
```

If the repo is private and the clone failed, clone it manually (`gh auth login` or a deploy key).
You can also rerun `sudo ./scripts/bootstrap_vm.sh $USER` at any time because it is idempotent.

## 2. Configure

```bash
cd ~/llm-serving-lab
make env            # creates compose/.env from compose/.env.example
nano compose/.env   # HF_TOKEN (only for gated models), MODEL_ID, image tags...
make venv           # Python venv for the benchmark harness + tests
```

T4 notes: keep `DTYPE=float16` because T4 has no bf16 support. `SGLANG_EXTRA_ARGS=--attention-backend triton`
is the most portable choice on sm75. On A100, use `DTYPE=bfloat16` and clear the SGLang extra args.

## 3. Start a backend + gateway

```bash
make up-vllm        # or up-sglang / up-tgi / up-ollama
make ps
make logs BACKEND=vllm
```

The first start downloads images (10–20 GB each) and model weights (~3 GB, shared HF cache
volume), so it can take several minutes. `--wait` blocks until the healthcheck passes
(start period up to 15 min).

`make up-all` starts all four backends on one GPU with the reduced memory fractions in
`compose/all.env`. Use it for gateway demos, **not** for benchmarking.

## 4. Smoke test

On the VM:

```bash
make smoke
```

From your laptop through an SSH tunnel:

```bash
./scripts/tunnel.sh azureuser@<fqdn>            # bash
.\scripts\tunnel.ps1 azureuser@<fqdn>           # PowerShell
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"vllm/qwen2.5-1.5b","messages":[{"role":"user","content":"Hi!"}]}'
```

Any OpenAI SDK works with `base_url="http://localhost:8000/v1"` and any `api_key`.

## 5. Benchmark

Full fair comparison: each backend runs alone on the GPU and is torn down before the next starts.

```bash
make bench-all
# customise
BACKENDS="vllm sglang" CONCURRENCY=1,4,16,64 MAX_TOKENS=256 BENCH_ARGS="--ignore-eos" \
  ./scripts/run_benchmark.sh
```

Output goes to `results/<timestamp>/`:

* `<backend>/summary.json`, `<backend>/requests.csv` contain raw per-level and per-request data
* `<backend>.server.log` and `<backend>.gpu-mem.txt` contain engine logs and GPU memory in use
* `report/report.md` and `report/*.png` contain the combined tables and charts

Single backend, already running: `make bench BACKEND=sglang`, then `make report`.
Direct backend, without the gateway (to measure gateway overhead):
`.venv/bin/python -m bench run --base-url http://127.0.0.1:8001/v1 --model Qwen/Qwen2.5-1.5B-Instruct --label vllm-direct --out results/manual/vllm-direct`

Copy results to your laptop: `scp -r azureuser@<fqdn>:~/llm-serving-lab/results ./results`.
Summarise findings with [docs/results/TEMPLATE.md](results/TEMPLATE.md).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit not configured: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Container says driver too old / `CUDA driver version is insufficient` | Image needs a newer driver: `sudo NVIDIA_DRIVER_BRANCH=580-server ./scripts/bootstrap_vm.sh $USER` then reboot, **or** pin an older image tag in `compose/.env`. |
| `nvidia-smi` fails after reboot | Check `mokutil --sb-state`; the Ubuntu-signed modules support Secure Boot, but custom/DKMS drivers do not. |
| vLLM: `Bfloat16 is only supported on GPUs with compute capability >= 8.0` | Set `DTYPE=float16`. |
| OOM / KV cache too small | Lower `MAX_MODEL_LEN`, or lower the memory fraction of *other* engines when using `up-all`. |
| SGLang kernel / FlashInfer errors on T4 | Keep `SGLANG_EXTRA_ARGS=--attention-backend triton`; try `--disable-cuda-graph` if capture fails. |
| TGI fails on T4 with flash-attention errors | Try `TGI_EXTRA_ARGS=--disable-custom-kernels` or a different TGI tag. Check `make logs BACKEND=tgi`. |
| Ollama answers slowly at first | The first request loads the model; the benchmark warmup covers this. `OLLAMA_KEEP_ALIVE=-1` keeps it loaded. |
| Gateway 502 `backend unreachable` | That backend's profile isn't running. Check `curl localhost:8000/health?refresh=true`. |
| Disk full | `docker system df`; remove unused images with `docker image prune -a`. |

## 7. Cost control and teardown

Approximate pay-as-you-go prices; check the [Azure pricing calculator](https://azure.microsoft.com/pricing/calculator/)
for your region.

| Size | GPU | ≈ $/hour (Linux, East US) |
|---|---|---|
| Standard_NC4as_T4_v3 | 1× T4 16 GB | ~0.5 |
| Standard_NC24ads_A100_v4 | 1× A100 80 GB | ~3.7 |

Spot is often 60–90% cheaper but can be evicted. Premium SSD and the public IP are billed
even while the VM is deallocated, at a few dollars per month.

```bash
./infra/deploy.sh stop    # deallocate: stops compute billing, keeps disk + caches
./infra/deploy.sh start   # resume later
./infra/deploy.sh down    # delete the resource group - stops all billing
```

The auto-shutdown schedule deallocates the VM daily at `AUTO_SHUTDOWN` (default 19:00 UTC) as a
safety net.
