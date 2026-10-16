#!/usr/bin/env bash
# Provision / inspect / delete the lab VM with Azure CLI + Bicep.
#
#   infra/deploy.sh quota   # show GPU vCPU quota in $LOCATION
#   infra/deploy.sh up      # create resource group + VM (prints ssh / tunnel commands)
#   infra/deploy.sh info    # print deployment outputs again
#   infra/deploy.sh stop    # deallocate VM (stops GPU billing; disk still billed)
#   infra/deploy.sh start   # start a deallocated VM
#   infra/deploy.sh down    # delete the whole resource group
#
# Settings (env vars):
#   RG=rg-llm-serving-lab LOCATION=eastus VM_SIZE=Standard_NC4as_T4_v3 NAME_PREFIX=llmlab
#   ADMIN_USER=azureuser SSH_KEY_FILE=~/.ssh/id_ed25519.pub SSH_SOURCE=<cidr, default: your IP/32>
#   REPO_URL=<git url cloned on the VM> AUTO_SHUTDOWN=1900 AUTO_SHUTDOWN_TZ=UTC USE_SPOT=false
set -euo pipefail

ACTION="${1:-up}"
RG="${RG:-rg-llm-serving-lab}"
LOCATION="${LOCATION:-eastus}"
VM_SIZE="${VM_SIZE:-Standard_NC4as_T4_v3}"
NAME_PREFIX="${NAME_PREFIX:-llmlab}"
ADMIN_USER="${ADMIN_USER:-azureuser}"
SSH_KEY_FILE="${SSH_KEY_FILE:-${HOME}/.ssh/id_ed25519.pub}"
SSH_SOURCE="${SSH_SOURCE:-}"
REPO_URL="${REPO_URL:-https://github.com/sagarsahu27/llm-serving-lab.git}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1900}"
AUTO_SHUTDOWN_TZ="${AUTO_SHUTDOWN_TZ:-UTC}"
USE_SPOT="${USE_SPOT:-false}"
DEPLOYMENT="llm-serving-lab"
HERE="$(cd "$(dirname "$0")" && pwd)"

die() { echo "error: $*" >&2; exit 1; }
command -v az >/dev/null || die "Azure CLI (az) not found"

outputs() {
  az deployment group show -g "${RG}" -n "${DEPLOYMENT}" \
    --query 'properties.outputs.{vm:vmName.value, ip:publicIp.value, fqdn:fqdn.value, ssh:sshCommand.value, tunnel:tunnelCommand.value}' \
    -o yaml
}

case "${ACTION}" in
  quota)
    az vm list-usage -l "${LOCATION}" \
      --query "[?contains(name.value, 'NCAS') || contains(name.value, 'NCADS') || name.value=='lowPriorityCores'].{family:localName, used:currentValue, limit:limit}" \
      -o table
    echo "NC4as_T4_v3 needs 4 vCPUs of 'Standard NCASv3_T4 Family'; NC24ads_A100_v4 needs 24 of 'Standard NCADS_A100_v4 Family'."
    ;;
  up)
    [[ -f "${SSH_KEY_FILE}" ]] || die "SSH public key ${SSH_KEY_FILE} not found (ssh-keygen -t ed25519)"
    if [[ -z "${SSH_SOURCE}" ]]; then
      my_ip="$(curl -fsS https://api.ipify.org)" || die "could not detect public IP; set SSH_SOURCE"
      SSH_SOURCE="${my_ip}/32"
    fi
    echo "Deploying ${VM_SIZE} in ${LOCATION} (rg ${RG}); SSH allowed from ${SSH_SOURCE}"
    az group create -n "${RG}" -l "${LOCATION}" -o none
    az deployment group create -g "${RG}" -n "${DEPLOYMENT}" \
      --template-file "${HERE}/main.bicep" \
      --parameters namePrefix="${NAME_PREFIX}" vmSize="${VM_SIZE}" adminUsername="${ADMIN_USER}" \
        sshPublicKey="$(cat "${SSH_KEY_FILE}")" allowedSshSourcePrefix="${SSH_SOURCE}" \
        repoUrl="${REPO_URL}" autoShutdownTime="${AUTO_SHUTDOWN}" \
        autoShutdownTimeZone="${AUTO_SHUTDOWN_TZ}" useSpot="${USE_SPOT}" \
      -o none
    outputs
    echo
    echo "First boot installs the NVIDIA driver + Docker and reboots (~10 min)."
    echo "Check progress: ssh in, then: tail -f /var/log/llm-serving-lab-bootstrap.log; cloud-init status --wait"
    ;;
  info)
    outputs
    ;;
  stop)
    az vm deallocate -g "${RG}" -n "${NAME_PREFIX}-vm"
    ;;
  start)
    az vm start -g "${RG}" -n "${NAME_PREFIX}-vm"
    outputs
    ;;
  down)
    read -r -p "Delete resource group ${RG} and everything in it? [y/N] " answer
    [[ "${answer}" =~ ^[Yy]$ ]] || exit 0
    az group delete -n "${RG}" --yes --no-wait
    echo "Deletion started."
    ;;
  *)
    die "unknown action ${ACTION} (quota|up|info|stop|start|down)"
    ;;
esac
