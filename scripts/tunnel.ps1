# Windows equivalent of scripts/tunnel.sh (uses the built-in OpenSSH client).
#   .\scripts\tunnel.ps1 azureuser@<vm-fqdn>
# Then: curl.exe http://localhost:8000/v1/models
param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$SshArgs
)
$ErrorActionPreference = 'Stop'
Write-Host 'Tunnelling gateway :8000, vLLM :8001, SGLang :8002, TGI :8003, Ollama :11434 (Ctrl+C to stop)'
$forwards = @(8000, 8001, 8002, 8003, 11434) | ForEach-Object { '-L', "${_}:127.0.0.1:${_}" }
& ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 @forwards @SshArgs $HostName
