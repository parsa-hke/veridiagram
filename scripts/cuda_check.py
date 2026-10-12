"""GPU check used by the lab setup scripts: PyTorch sees CUDA, every GPU listed, bf16 works."""

import sys

import torch

print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| available", torch.cuda.is_available(),
      "| GPUs", torch.cuda.device_count())
if not torch.cuda.is_available():
    print("CUDA is NOT available to PyTorch. The PyTorch build probably needs a newer NVIDIA driver; rerun the "
          "setup with an older CUDA build, e.g. TORCH_INDEX=https://download.pytorch.org/whl/cu126")
    sys.exit(1)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory / 2**30:.0f} GB, bf16 "
          f"{'yes' if torch.cuda.is_bf16_supported() else 'NO (add --set finetune.amp_dtype=float16)'}")
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", tuple((x @ x).shape))
