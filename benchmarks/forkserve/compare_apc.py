#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare APC vs ForkServe live KV pages on agent-style fan-out.

Runs against the in-process V1 KVCacheManager (CPU). No GPU required.

    python benchmarks/forkserve/compare_apc.py
    python benchmarks/forkserve/compare_apc.py --json /tmp/forkserve_apc.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.forkserve import (
    allocated_slots,
    forkserve_extra,
    live_block_ids,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request

# Llama-3-8B GQA bf16: 2 * 32 layers * 8 kv heads * 128 * 2 bytes
BYTES_PER_TOKEN_8B = 2 * 32 * 8 * 128 * 2


@dataclass
class Row:
    name: str
    L: int
    k: int
    ell: int
    page: int
    apc_live_blocks: int
    fork_live_blocks: int
    apc_formula_slots: int
    fork_formula_slots: int
    saving_vs_apc: float
    hash_bypasses: int


def _manager(page: int, num_blocks: int) -> KVCacheManager:
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=page,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        cfg,
        max_model_len=max(32768, page * 64),
        scheduler_block_size=page,
        hash_block_size=page,
        enable_caching=True,
    )


def _req(rid: str, tokens: list[int], page: int, extra: dict | None = None) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1, extra_args=extra),
        pooling_params=None,
        block_hasher=get_request_block_hasher(page, sha256),
    )


def _alloc(mgr: KVCacheManager, req: Request) -> None:
    computed, n_hit, _ = mgr.get_computed_blocks(req)
    assert mgr.allocate_slots(req, req.num_tokens - n_hit, n_hit, computed) is not None


def run_apc_same_batch(L: int, k: int, ell: int, page: int) -> int:
    """All children look up before any hash is published → APC miss / clone."""
    mgr = _manager(page, num_blocks=max(2048, k * ((L + ell) // page + 8)))
    trunk = list(range(L))
    reqs = [_req(f"a{i}", trunk + [10_000 + i] * ell, page) for i in range(k)]
    hits = [mgr.get_computed_blocks(r) for r in reqs]
    for req, (blocks, n_hit, _) in zip(reqs, hits):
        assert mgr.allocate_slots(req, req.num_tokens - n_hit, n_hit, blocks)
    return len(live_block_ids(mgr.block_pool))


def run_forkserve(L: int, k: int, ell: int, page: int) -> tuple[int, int]:
    mgr = _manager(page, num_blocks=max(2048, (L // page) + k + 8))
    trunk = list(range(L))
    parent = _req("p", trunk, page, forkserve_extra(node=1, session="bench"))
    _alloc(mgr, parent)
    for i in range(k):
        child = _req(
            f"c{i}",
            trunk + [10_000 + i] * ell,
            page,
            forkserve_extra(node=2 + i, parent=1, session="bench"),
        )
        _alloc(mgr, child)
    return len(live_block_ids(mgr.block_pool)), mgr.forkserve.stats.hash_bypasses


SCENARIOS = [
    ("aligned_fanout_4", 2048, 4, 64, 16),
    ("aligned_fanout_8", 4096, 8, 128, 16),
    ("unaligned_pack", 2050, 4, 48, 16),
    ("tot_wide", 8192, 8, 32, 16),
    ("short_residual", 1024, 6, 4, 16),
    ("planner_specialists", 16384, 4, 256, 16),
]


def run_all() -> list[Row]:
    init_none_hash(sha256)
    rows: list[Row] = []
    for name, L, k, ell, page in SCENARIOS:
        apc = run_apc_same_batch(L, k, ell, page)
        fork, bypass = run_forkserve(L, k, ell, page)
        apc_slots = allocated_slots(L=L, k=k, ell=ell, page=page, scheme="clone")
        fork_slots = allocated_slots(L=L, k=k, ell=ell, page=page, scheme="pack")
        saving = 0.0 if apc == 0 else 1.0 - fork / apc
        rows.append(
            Row(
                name=name,
                L=L,
                k=k,
                ell=ell,
                page=page,
                apc_live_blocks=apc,
                fork_live_blocks=fork,
                apc_formula_slots=apc_slots,
                fork_formula_slots=fork_slots,
                saving_vs_apc=saving,
                hash_bypasses=bypass,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()
    rows = run_all()
    print(
        f"{'scenario':<22} {'L':>6} {'k':>3} {'ell':>5} "
        f"{'APC blk':>8} {'FS blk':>8} {'save':>7} "
        f"{'APC GiB*':>9} {'FS GiB*':>9}"
    )
    print("-" * 90)
    for r in rows:
        apc_gib = r.apc_formula_slots * BYTES_PER_TOKEN_8B / 1024**3
        fs_gib = r.fork_formula_slots * BYTES_PER_TOKEN_8B / 1024**3
        print(
            f"{r.name:<22} {r.L:>6} {r.k:>3} {r.ell:>5} "
            f"{r.apc_live_blocks:>8} {r.fork_live_blocks:>8} "
            f"{100 * r.saving_vs_apc:>6.1f}% "
            f"{apc_gib:>9.2f} {fs_gib:>9.2f}"
        )
    print()
    print("* GiB is Llama-3-8B bf16 closed-form allocated slots, not the")
    print("  unit-test block pool (1 dummy layer). Live blk is the pool.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
