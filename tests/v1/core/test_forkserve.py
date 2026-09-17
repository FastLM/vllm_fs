# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ForkServe vs APC: CoW fork alias, row-level copy, storage comparison."""

from collections.abc import Callable

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.forkserve import (
    allocated_slots,
    choose_adaptive_tail,
    forkserve_extra,
    live_block_ids,
    pages_freeze,
    pages_pack,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.request import Request
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace, cow_copy_kv_rows

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _hash():
    init_none_hash(sha256)


def _make_request(
    request_id: str,
    tokens: list[int],
    block_size: int,
    extra: dict | None = None,
) -> Request:
    params = SamplingParams(max_tokens=1, extra_args=extra)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        block_hasher=None,
    )


def _manager(block_size: int = 16, num_blocks: int = 256) -> KVCacheManager:
    cfg = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    mgr = KVCacheManager(
        cfg,
        max_model_len=8192,
        scheduler_block_size=block_size,
        hash_block_size=block_size,
        enable_caching=True,
    )
    return mgr


def _hasher(block_size: int) -> Callable:
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher

    return get_request_block_hasher(block_size, sha256)


def _req(
    rid: str,
    tokens: list[int],
    block_size: int,
    extra: dict | None = None,
) -> Request:
    params = SamplingParams(max_tokens=1, extra_args=extra)
    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        block_hasher=_hasher(block_size),
    )


def _allocate_miss(mgr: KVCacheManager, req: Request) -> None:
    computed, n_hit, _ = mgr.get_computed_blocks(req)
    n_new = req.num_tokens - n_hit
    assert mgr.allocate_slots(req, n_new, n_hit, computed) is not None
    req.num_computed_tokens = req.num_tokens


def test_cow_copy_kv_rows_copies_prefix_only():
    block_size = 4
    blocks = torch.zeros(2, 2, block_size, 3)
    blocks[0, 0, :, :] = torch.arange(block_size * 3).reshape(block_size, 3)
    blocks[0, 1, :, :] = 7
    cow_copy_kv_rows(blocks, 0, 1, n_valid=2, block_size=block_size)
    torch.testing.assert_close(blocks[1, :, :2], blocks[0, :, :2])
    assert (blocks[1, :, 2:] == 0).all()


def test_copy_kv_cache_blocks_respects_n_valid():
    num_blocks = 4
    block_size = 8
    cache = torch.zeros(num_blocks, 2, block_size, 2)
    cache[0] = 3
    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks,
        [KVCacheBlockCopy(0, 2, n_valid=3, block_size=block_size)],
    )
    torch.testing.assert_close(cache[2, :, :3], cache[0, :, :3])
    assert (cache[2, :, 3:] == 0).all()


def test_full_block_copy_unchanged():
    num_blocks = 4
    cache = torch.zeros(num_blocks, 2, 2)
    cache[1] = 9
    copy_kv_cache_blocks_inplace(
        [cache],
        num_blocks,
        [KVCacheBlockCopy(1, 0)],
    )
    torch.testing.assert_close(cache[0], cache[1])


def test_fork_aliases_trunk_without_hash_walk():
    P, L, ell, k = 16, 64, 8, 4
    trunk = list(range(L))
    mgr = _manager(P, num_blocks=128)
    parent = _req("p", trunk, P, forkserve_extra(node=1, session="s"))
    _allocate_miss(mgr, parent)
    live_after_parent = len(live_block_ids(mgr.block_pool))

    for i in range(k):
        child_toks = trunk + [1000 + i] * ell
        child = _req(
            f"c{i}",
            child_toks,
            P,
            forkserve_extra(node=10 + i, parent=1, session="s"),
        )
        _allocate_miss(mgr, child)

    assert mgr.forkserve.stats.hash_bypasses == k
    live = len(live_block_ids(mgr.block_pool))
    # One trunk plus k residual pages (ell < P).
    assert live == live_after_parent + k
    assert live < k * ((L + ell + P - 1) // P)


def test_apc_same_batch_miss_clones_trunk():
    P, L, ell, k = 16, 64, 8, 4
    mgr = _manager(P, num_blocks=256)
    prompts = [list(range(L)) + [1000 + i] * ell for i in range(k)]
    reqs = [_req(f"a{i}", prompts[i], P) for i in range(k)]
    hits = [mgr.get_computed_blocks(r) for r in reqs]
    assert all(n == 0 for _, n, _ in hits)
    for req, (blocks, n_hit, _) in zip(reqs, hits):
        n_new = req.num_tokens - n_hit
        assert mgr.allocate_slots(req, n_new, n_hit, blocks) is not None
    live = len(live_block_ids(mgr.block_pool))
    # k independent trunks (no hash published before allocate of the batch).
    assert live >= k * (L // P)


def test_fork_beats_apc_same_batch_on_live_blocks():
    P, L, ell, k = 16, 128, 16, 6
    trunk = list(range(L))

    apc = _manager(P, num_blocks=512)
    prompts = [trunk + [2000 + i] * ell for i in range(k)]
    reqs = [_req(f"a{i}", prompts[i], P) for i in range(k)]
    hits = [apc.get_computed_blocks(r) for r in reqs]
    for req, (blocks, n_hit, _) in zip(reqs, hits):
        assert apc.allocate_slots(req, req.num_tokens - n_hit, n_hit, blocks)
    apc_live = len(live_block_ids(apc.block_pool))

    fs = _manager(P, num_blocks=512)
    parent = _req("p", trunk, P, forkserve_extra(node=1, session="t"))
    _allocate_miss(fs, parent)
    for i in range(k):
        child = _req(
            f"c{i}",
            trunk + [2000 + i] * ell,
            P,
            forkserve_extra(node=2 + i, parent=1, session="t"),
        )
        _allocate_miss(fs, child)
    fs_live = len(live_block_ids(fs.block_pool))

    assert fs_live < apc_live
    # Aligned trunk:  L/P shared + k residual pages vs k * (L+ell)/P.
    assert fs_live == L // P + k
    assert apc_live == k * ((L + ell) // P)


def test_abort_child_drops_residual_keeps_trunk():
    P, L, ell = 16, 48, 4
    trunk = list(range(L))
    mgr = _manager(P)
    parent = _req("p", trunk, P, forkserve_extra(node=1, session="s"))
    _allocate_miss(mgr, parent)
    child = _req(
        "c",
        trunk + [9] * ell,
        P,
        forkserve_extra(node=2, parent=1, session="s"),
    )
    _allocate_miss(mgr, child)
    live_both = len(live_block_ids(mgr.block_pool))
    mgr.free(child)
    live_after = len(live_block_ids(mgr.block_pool))
    assert live_after == live_both - 1
    assert live_after == L // P


def test_parent_free_keeps_pinned_trunk_for_children():
    P, L, ell = 16, 32, 8
    trunk = list(range(L))
    mgr = _manager(P)
    parent = _req("p", trunk, P, forkserve_extra(node=1, session="s"))
    _allocate_miss(mgr, parent)
    child = _req(
        "c",
        trunk + [3] * ell,
        P,
        forkserve_extra(node=2, parent=1, session="s"),
    )
    _allocate_miss(mgr, child)
    mgr.free(parent)
    assert len(live_block_ids(mgr.block_pool)) == L // P + 1
    mgr.free(child)
    assert len(live_block_ids(mgr.block_pool)) == 0


def test_unaligned_pack_matches_formula():
    P, L, ell, k = 16, 50, 20, 3
    r = L % P
    trunk = list(range(L))
    mgr = _manager(P, num_blocks=256)
    parent = _req("p", trunk, P, forkserve_extra(node=1, session="u"))
    _allocate_miss(mgr, parent)
    for i in range(k):
        child = _req(
            f"c{i}",
            trunk + [8] * ell,
            P,
            forkserve_extra(node=2 + i, parent=1, session="u"),
        )
        _allocate_miss(mgr, child)
    live = len(live_block_ids(mgr.block_pool))
    # Parent keeps its partial tail page; children pack r+ell privately.
    expected = (L + P - 1) // P + pages_pack(k, r, ell, P)
    assert live == expected


def test_adaptive_tail_chooses_pack_when_residual_fits():
    assert choose_adaptive_tail(k=4, r=15, ell=1, page=16) == "pack"
    assert choose_adaptive_tail(k=4, r=15, ell=128, page=16) == "freeze"
    assert choose_adaptive_tail(k=4, r=0, ell=16, page=16) == "aligned"


def test_allocated_slots_clone_vs_fork():
    L, k, ell, P = 8192, 4, 128, 16
    clone = allocated_slots(L=L, k=k, ell=ell, page=P, scheme="clone")
    fork = allocated_slots(L=L, k=k, ell=ell, page=P, scheme="adaptive")
    assert fork < clone
    assert fork == L + k * ell
    assert clone == k * (L + ell)


def test_block_not_writable_when_ro():
    mgr = _manager()
    block = mgr.block_pool.blocks[1]
    block.ref_cnt = 1
    assert mgr.block_pool.is_block_writable(block)
    block.ro = True
    assert not mgr.block_pool.is_block_writable(block)
