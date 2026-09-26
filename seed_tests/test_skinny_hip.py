"""CPU checks for `skinny_hip` routing (the kernel itself is checked on MI300A by
`scratchpad/skinny_hip_bench.py`, which compares every config against `F.linear`)."""

from __future__ import annotations

import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import skinny_hip  # noqa: E402


def test_configs_mirror_hip_table():
    src = skinny_hip.HIP_SRC.read_text()
    table = src.split("#define SK_CONFIGS(X)", 1)[1].split("extern", 1)[0]
    rows = re.findall(r"X\((\d+), (\d+), (\d+), (\d+), (\d+)\)", table)
    assert [tuple(int(v) for v in r[1:]) for r in sorted(rows, key=lambda r: int(r[0]))] == [
        tuple(c) for c in skinny_hip.CONFIGS
    ]
    assert [int(r[0]) for r in rows] == list(range(len(rows)))


def test_routes_are_valid_configs():
    for (n, k, b), (cfg, tpp) in skinny_hip.ROUTES.items():
        assert b in skinny_hip.BUCKETS
        for m in range(1, b + 1):
            if skinny_hip.bucket(m) == b and skinny_hip.mt_of(m) == skinny_hip.mt_of(b):
                assert skinny_hip.valid(cfg, m, n, k, tpp), (n, k, b, cfg, m)
        # every M in the bucket needs the config's M tile count
        lo = max(bb for bb in (0, *skinny_hip.BUCKETS) if bb < b) + 1
        assert skinny_hip.mt_of(lo) == skinny_hip.mt_of(b) == skinny_hip.CONFIGS[cfg][0]


def test_bucket():
    assert [skinny_hip.bucket(m) for m in (1, 2, 16, 17, 32, 33, 64)] == [1, 16, 16, 32, 32, 48, 64]


def test_linear_falls_back_off_gpu():
    x = torch.randn(3, 1, 256, dtype=torch.bfloat16)
    w = torch.randn(4096, 256, dtype=torch.bfloat16)
    prev = skinny_hip.ENABLED
    try:
        for flag in (False, True):
            skinny_hip.ENABLED = flag
            assert torch.equal(skinny_hip.linear(x, w), F.linear(x, w))
    finally:
        skinny_hip.ENABLED = prev
