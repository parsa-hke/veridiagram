# One-time setup on a Windows GPU machine (UOWD AI Lab; LAB_GUIDE.md). From the repository folder:
#   powershell -ExecutionPolicy Bypass -File scripts\lab_setup.ps1
# Installs uv, Python 3.11 and the pinned packages into .venv (no admin rights), PyTorch from the
# CUDA index that matches the NVIDIA driver (PyTorch's Windows wheels on PyPI are CPU-only),
# checks the GPUs and runs the tests. Force a build with: $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
$ErrorActionPreference = "Continue"   # native tools (uv, pytest) write progress to stderr; exit codes are checked instead
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "== 1/5 GPUs and driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi not found: this machine has no NVIDIA driver. Ask the lab staff."; exit 1
}
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
$driverStr = (nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
$driver = [int]($driverStr.Split(".")[0])

Write-Host "== 2/5 uv"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version
if ($LASTEXITCODE -ne 0) { Write-Host "uv is not available"; exit 1 }

Write-Host "== 3/5 Python 3.11 and pinned packages"
uv python install 3.11
if (-not (Test-Path .venv)) { uv venv --python 3.11 .venv }
$py = ".\.venv\Scripts\python.exe"
if (-not $env:TORCH_INDEX) {
    if     ($driver -ge 580) { $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu130" }
    elseif ($driver -ge 570) { $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu128" }
    elseif ($driver -ge 560) { $env:TORCH_INDEX = "https://download.pytorch.org/whl/cu126" }
    else { Write-Host "Driver $driverStr is older than 560: ask the lab staff to update the NVIDIA driver."; exit 1 }
}
Write-Host "driver $driverStr -> PyTorch from $env:TORCH_INDEX"
uv pip install --python $py torch==2.14.1 torchvision==0.29.1 --index-url $env:TORCH_INDEX
if ($LASTEXITCODE -ne 0) { Write-Host "PyTorch install failed. Try another CUDA build, e.g. cu126 (see the header)."; exit 1 }
$req = Join-Path $env:TEMP "veridiagram_req_no_torch.txt"
Get-Content requirements.txt | Where-Object { $_ -notmatch '^(torch|torchvision)==' } | Set-Content $req
uv pip install --python $py -r $req
if ($LASTEXITCODE -ne 0) { Write-Host "package install failed"; exit 1 }

Write-Host "== 4/5 CUDA check"
& $py scripts\cuda_check.py
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== 5/5 tests (no downloads)"
& $py -m pytest -q 2>&1 | Select-Object -Last 1
Write-Host "setup done"
