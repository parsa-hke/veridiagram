#!/usr/bin/env bash
# One-time setup on a Linux GPU machine (UOWD AI Lab; plan §0.6, LAB_GUIDE.md).
#   bash scripts/lab_setup.sh
# Installs uv, Python 3.11 and the pinned packages into .venv (no admin rights needed), picks a
# PyTorch CUDA build the NVIDIA driver supports, checks the GPUs and runs the tests.
# Override the PyTorch build with TORCH_INDEX=https://download.pytorch.org/whl/cuXXX bash scripts/lab_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1/5 GPUs and driver"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found: this machine has no NVIDIA driver (or it is not on PATH). Ask the lab staff."; exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)

echo "== 2/5 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user --quiet uv 2>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

echo "== 3/5 Python 3.11 and pinned packages"
uv python install 3.11
[ -d .venv ] || uv venv --python 3.11 .venv
if [ -z "${TORCH_INDEX:-}" ]; then
  # Default PyPI wheel of torch 2.14 is built for CUDA 13.0 (needs driver >= 580).
  if   [ "$DRIVER" -ge 580 ]; then TORCH_INDEX=""
  elif [ "$DRIVER" -ge 570 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu128"
  elif [ "$DRIVER" -ge 560 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  else echo "Driver $DRIVER is older than 560: ask the lab staff, or find a matching build at https://pytorch.org/get-started/previous-versions/ and rerun with TORCH_INDEX=..."; exit 1
  fi
fi
if [ -n "$TORCH_INDEX" ]; then
  echo "driver $DRIVER: installing torch from $TORCH_INDEX"
  uv pip install --python .venv/bin/python torch==2.14.1 torchvision==0.29.1 --index-url "$TORCH_INDEX"
fi
uv pip install --python .venv/bin/python -r requirements.txt

echo "== 4/5 CUDA check"
.venv/bin/python - <<'PY'
import sys
import torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| available", torch.cuda.is_available(),
      "| GPUs", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("CUDA is NOT available to PyTorch. The build probably needs a newer driver: rerun with\n"
          "  TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/lab_setup.sh")
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GB, "
          f"bf16 {'yes' if torch.cuda.is_bf16_supported() else 'NO (use finetune.amp_dtype=float16)'}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", tuple((x @ x).shape))
PY

echo "== 5/5 tests (no downloads)"
.venv/bin/python -m pytest -q 2>&1 | tail -1
echo "setup done"
