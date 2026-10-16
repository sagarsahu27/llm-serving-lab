#!/usr/bin/env bash
# Prepare an Ubuntu 22.04 Azure GPU VM for the lab:
#   NVIDIA datacenter driver (Ubuntu-signed, Secure Boot friendly), Docker Engine +
#   compose plugin, NVIDIA Container Toolkit, Python venv tooling, optional repo clone.
#
# Idempotent. Used by infra/cloud-init.yaml on first boot, or run manually:
#   sudo ./scripts/bootstrap_vm.sh <username>
#
# Env:
#   NVIDIA_DRIVER_BRANCH  driver branch to install (default 580-server; falls back to the
#                         ubuntu-drivers recommendation if unavailable). Recent vLLM/SGLang
#                         images are built on CUDA 12.8+/13 and need a >=570/580 driver.
#   REPO_URL              git URL to clone into ~<username>/llm-serving-lab (optional)
set -euo pipefail

TARGET_USER="${1:-${SUDO_USER:-}}"
NVIDIA_DRIVER_BRANCH="${NVIDIA_DRIVER_BRANCH:-580-server}"
REPO_URL="${REPO_URL:-}"
export DEBIAN_FRONTEND=noninteractive

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

if [[ ${EUID} -ne 0 ]]; then
  echo "must run as root (sudo $0 <username>)" >&2
  exit 1
fi

log "installing base packages"
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg git make jq python3-venv python3-pip ubuntu-drivers-common

# --- NVIDIA driver ----------------------------------------------------------------------
reboot_required=0
if nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA driver already active: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
else
  if apt-cache show "nvidia-headless-no-dkms-${NVIDIA_DRIVER_BRANCH}" >/dev/null 2>&1; then
    log "installing NVIDIA driver branch ${NVIDIA_DRIVER_BRANCH}"
    ubuntu-drivers install --gpgpu "nvidia:${NVIDIA_DRIVER_BRANCH}"
  else
    log "driver branch ${NVIDIA_DRIVER_BRANCH} not available; using ubuntu-drivers recommendation"
    ubuntu-drivers install --gpgpu
  fi
  # --gpgpu installs the headless driver only; nvidia-smi lives in nvidia-utils-<branch>.
  branch="$(dpkg-query -W -f='${Package}\n' 'nvidia-headless-no-dkms-*' 2>/dev/null \
    | sed -n 's/^nvidia-headless-no-dkms-//p' | sort -V | tail -n1)"
  if [[ -n "${branch}" ]]; then
    apt-get install -y "nvidia-utils-${branch}"
  fi
  reboot_required=1
fi

# --- Docker Engine ----------------------------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  log "installing Docker Engine"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  # shellcheck disable=SC1091
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${codename} stable" \
    >/etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

# --- NVIDIA Container Toolkit -----------------------------------------------------------
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  log "installing NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker

# --- User + repo ------------------------------------------------------------------------
if [[ -n "${TARGET_USER}" ]] && id "${TARGET_USER}" >/dev/null 2>&1; then
  usermod -aG docker "${TARGET_USER}"
  home_dir="$(getent passwd "${TARGET_USER}" | cut -d: -f6)"
  if [[ -n "${REPO_URL}" && ! -d "${home_dir}/llm-serving-lab" ]]; then
    log "cloning ${REPO_URL}"
    sudo -u "${TARGET_USER}" git clone "${REPO_URL}" "${home_dir}/llm-serving-lab" \
      || log "clone failed (private repo?) - clone manually after login"
  fi
fi

mkdir -p /var/lib/llm-serving-lab
date -u +%FT%TZ >/var/lib/llm-serving-lab/bootstrap.done
if [[ ${reboot_required} -eq 1 ]]; then
  log "done - REBOOT REQUIRED to load the NVIDIA driver (cloud-init reboots automatically)"
else
  log "done"
fi
