"""Query as token indices instead of a highlight overlay (plan §7.8 ablation).

A query is encoded as N_SLOTS integer tokens: slot 0 is the query kind, then two name slots per ref
(up to three refs): a segment gives its two endpoint names, a point its name, a circle its centre's
name (or the CIRCLE token when the centre is unnamed). Point names are mapped to ids by a stable hash
(crc32) into NAME_BUCKETS buckets, so a name is the same token in every diagram (the model has to
find the letter in the image). 0 is padding.
"""

from __future__ import annotations

import zlib

import torch

KINDS = ("seg_pair", "pt_circle", "seg_circle", "pt_line")
N_SLOTS = 7
NAME_BUCKETS = 1024
CIRCLE = NAME_BUCKETS + 1
KIND_OFFSET = NAME_BUCKETS + 2
VOCAB_SIZE = KIND_OFFSET + len(KINDS)


def name_token(name: str) -> int:
    return 1 + zlib.crc32(str(name).encode()) % NAME_BUCKETS


def encode(kind: str, refs: list, circle_centres: dict | None = None) -> torch.Tensor:
    """(N_SLOTS,) long tensor. `circle_centres` maps circle index -> centre point name (or None)."""
    toks = [KIND_OFFSET + KINDS.index(kind)]
    for ref in refs[:3]:
        t = ref[0]
        if t == "seg":
            names = [name_token(ref[1]), name_token(ref[2])]
        elif t == "pt":
            names = [name_token(ref[1]), 0]
        elif t == "circle":
            c = (circle_centres or {}).get(int(ref[1]))
            names = [name_token(c) if c else CIRCLE, 0]
        else:
            raise ValueError(f"unknown ref type {t!r}")
        toks += names
    toks += [0] * (N_SLOTS - len(toks))
    return torch.tensor(toks, dtype=torch.long)
