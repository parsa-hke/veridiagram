"""DINOv3 ViT-L/16 (fallback DINOv2-L/14) loading and feature extraction (plan §6.1).

`Backbone(x)` returns a dict with
  cls      (B, D)          CLS token after the final norm
  patches  (B, N, D)       patch tokens (register tokens dropped)
  pooled   (B, 2D)         CLS concatenated with the mean of the patch tokens
Any input size works: DINOv3 uses RoPE (sizes must be multiples of 16, which the dataset pads to);
DINOv2 interpolates its position embeddings and ignores a remainder smaller than its 14 px patch.
Gradient checkpointing is switched on when `model.grad_checkpointing` is true.

`layer_depth(name)` maps a parameter name to its depth for layer-wise LR decay: embeddings 0,
transformer block i -> i + 1, the final norm num_layers + 1. Parameter names: DINOv3
`embeddings.*`, `model.layer.{i}.*`, `norm.*`; DINOv2 `embeddings.*`, `encoder.layer.{i}.*`,
`layernorm.*`.

`load_backbone(cfg)` tries `model.backbone`, falls back to `model.backbone_fallback` if that fails
(gated checkpoint without access, offline, ...), and records which one was loaded. It also checks the
checkpoint's image processor mean/std against `input.mean` / `input.std`.
"""

from __future__ import annotations

import re
import warnings

import torch
from torch import nn

LAYER_RE = re.compile(r"(?:^|\.)layer\.(\d+)\.")


class Backbone(nn.Module):
    def __init__(self, hf_model: nn.Module, name: str = ""):
        super().__init__()
        self.model = hf_model
        self.name = name
        cfg = hf_model.config
        self.dim = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.patch_size = cfg.patch_size

    @property
    def out_dim(self) -> int:
        return 2 * self.dim

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> dict:
        out = self.model(pixel_values=x, output_attentions=output_attentions)
        h = out.last_hidden_state
        cls, patches = h[:, 0], h[:, self.num_prefix:]
        res = {"cls": cls, "patches": patches, "pooled": torch.cat([cls, patches.mean(1)], dim=-1)}
        if output_attentions:
            res["attentions"] = out.attentions
        return res

    def layer_depth(self, param_name: str) -> int:
        m = LAYER_RE.search(param_name)
        if m:
            return int(m.group(1)) + 1
        if param_name.startswith("embeddings.") or ".embeddings." in param_name or "rope" in param_name:
            return 0
        return self.num_layers + 1                     # final norm and anything after the blocks


def _from_pretrained(name: str, grad_ckpt: bool) -> nn.Module:
    from transformers import AutoModel

    m = AutoModel.from_pretrained(name)
    if grad_ckpt and getattr(m, "supports_gradient_checkpointing", False):
        m.gradient_checkpointing_enable()
    return m


def check_normalisation(name: str, mean, std, tol: float = 1e-3) -> dict:
    """Compare the checkpoint's preprocessing mean/std with the config's."""
    try:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(name)
        pm, ps = list(proc.image_mean), list(proc.image_std)
    except Exception as e:  # noqa: BLE001 - informational
        return {"checked": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    ok = all(abs(a - b) <= tol for a, b in zip(pm + ps, list(mean) + list(std)))
    if not ok:
        warnings.warn(f"{name} expects mean {pm} std {ps}, config has {list(mean)} {list(std)}")
    return {"checked": True, "matches": ok, "processor_mean": pm, "processor_std": ps}


def load_backbone(cfg: dict) -> Backbone:
    mcfg = cfg["model"]
    tried = []
    for name in (mcfg["backbone"], mcfg.get("backbone_fallback")):
        if not name:
            continue
        try:
            m = _from_pretrained(name, mcfg.get("grad_checkpointing", True))
            bb = Backbone(m, name)
            bb.load_info = {"loaded": name, "tried": tried,
                            "normalisation": check_normalisation(name, cfg["input"]["mean"], cfg["input"]["std"])}
            if tried:
                warnings.warn(f"backbone {mcfg['backbone']} unavailable ({tried[0][1]}); using fallback {name}")
            return bb
        except Exception as e:  # noqa: BLE001 - try the fallback
            tried.append((name, f"{type(e).__name__}: {str(e)[:300]}"))
    raise RuntimeError(f"no backbone could be loaded: {tried}")


def tiny_backbone(kind: str = "dinov3", hidden: int = 32, layers: int = 2) -> Backbone:
    """Randomly initialised small model for tests (no download)."""
    if kind == "dinov3":
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        m = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=2,
                                           intermediate_size=2 * hidden, num_register_tokens=4, patch_size=16))
    else:
        from transformers import Dinov2Config, Dinov2Model
        m = Dinov2Model(Dinov2Config(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=2,
                                     intermediate_size=2 * hidden, patch_size=14, image_size=64))
    return Backbone(m, f"tiny-{kind}")


def describe(bb: Backbone, size: tuple[int, int] = (448, 320)) -> dict:
    """Architecture facts and a forward pass on a dummy input (used by the access check)."""
    c = bb.model.config
    x = torch.zeros(1, 3, *size)
    with torch.no_grad():
        out = bb(x)
    n_patches = (size[0] // bb.patch_size) * (size[1] // bb.patch_size)
    return {"name": bb.name, "class": type(bb.model).__name__, "hidden": bb.dim, "layers": bb.num_layers,
            "patch": bb.patch_size, "register_tokens": bb.num_prefix - 1,
            "params_M": round(sum(p.numel() for p in bb.parameters()) / 1e6, 1),
            "input": list(size), "patch_tokens": int(out["patches"].shape[1]), "expected_patch_tokens": n_patches,
            "pooled_dim": int(out["pooled"].shape[1]), "image_size_cfg": getattr(c, "image_size", None)}


def main(argv: list[str] | None = None) -> int:
    """python -m models.backbone [--config X]: load the PRIMARY backbone only (no fallback), run a
    forward pass and print its architecture and normalisation. Exits non-zero if it cannot be loaded,
    e.g. when the gated DINOv3 checkpoint is not accessible with the current Hugging Face token."""
    import argparse
    import json
    import os

    from experiments.run import REPO_ROOT, load_config

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    name = cfg["model"]["backbone"]
    token = "HF_TOKEN env" if os.environ.get("HF_TOKEN") else "cached login" if os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token")) else "none"
    print(f"loading {name} (no fallback); Hugging Face credentials: {token}", flush=True)
    try:
        bb = Backbone(_from_pretrained(name, grad_ckpt=False), name).eval()
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to load {name}: {type(e).__name__}: {str(e)[:300]}")
        print("If this is a 401/gated error: accept the licence on the model page, then set HF_TOKEN "
              "(Colab secret) or run `hf auth login`.")
        return 1
    info = describe(bb)
    info["normalisation"] = check_normalisation(name, cfg["input"]["mean"], cfg["input"]["std"])
    print(json.dumps(info, indent=2))
    ok = info["patch_tokens"] == info["expected_patch_tokens"] and info["normalisation"].get("matches", False)
    print(f"{name}: {'OK' if ok else 'CHECK FAILED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
