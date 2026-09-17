# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ForkServe: branch-aware CoW KV on top of vLLM V1's block pool.

APC discovers sharing *after* tokens exist by hashing full blocks. ForkServe
aliases a parent's page table *before* the residual exists, so a k-way fan-out
stores one trunk plus k residuals instead of k trunks.

Production path is pack (copy the unaligned tail into each child's first
residual page). Freeze-tail needs a fork-aware attention slot map; the
accounting helpers still report it so experiments can compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_utils import KVCacheBlock


def live_block_ids(pool: Any) -> set[int]:
    return {b.block_id for b in pool.blocks if b.ref_cnt > 0 and not b.is_null}

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.request import Request


def forkserve_extra(
    *,
    node: int,
    parent: int | None = None,
    session: str | None = None,
    speculative: bool = False,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "forkserve_node": int(node),
        "forkserve_class": "speculative" if speculative else "committed",
    }
    if parent is not None:
        extra["forkserve_parent"] = int(parent)
    if session:
        extra["forkserve_session"] = str(session)
    return extra


def extra_of(request: Request) -> dict[str, Any]:
    sp = getattr(request, "sampling_params", None)
    if sp is None:
        return {}
    return dict(getattr(sp, "extra_args", None) or {})


def node_key(session: Any, node: Any) -> str | None:
    if node is None:
        return None
    return f"{session or ''}:{int(node)}"


def select_full_blocks(
    blocks: list[KVCacheBlock],
    block_size: int,
    parent_tokens: int,
) -> tuple[list[KVCacheBlock], int]:
    """Keep complete pages only. The unaligned tail stays with the parent."""
    if block_size <= 0 or parent_tokens <= 0 or not blocks:
        return [], 0
    n_full = parent_tokens // block_size
    if n_full <= 0:
        return [], 0
    use = list(blocks[:n_full])
    return use, n_full * block_size


def pages_pack(k: int, r: int, ell: int, page: int) -> int:
    """Private pages if each child clones the unaligned tail into its residual."""
    if page <= 0:
        return 0
    return k * ((r + ell + page - 1) // page)


def pages_freeze(k: int, r: int, ell: int, page: int) -> int:
    """Private pages if the unaligned tail is one shared RO frame."""
    if page <= 0:
        return 0
    frozen = 1 if r else 0
    return frozen + k * ((ell + page - 1) // page)


def choose_adaptive_tail(k: int, r: int, ell: int, page: int) -> str:
    if r == 0:
        return "aligned"
    return "freeze" if pages_freeze(k, r, ell, page) < pages_pack(k, r, ell, page) else "pack"


def allocated_slots(
    *,
    L: int,
    k: int,
    ell: int,
    page: int,
    scheme: str,
) -> int:
    """Token-slots occupied by distinct physical pages."""
    r = L % page
    full_trunk = ((L - r) // page) * page
    if scheme in ("clone", "apc_miss"):
        return k * ((L + ell + page - 1) // page) * page
    if scheme in ("apc_hit", "extra_pin", "pack"):
        return full_trunk + k * ((r + ell + page - 1) // page) * page
    if scheme == "freeze":
        return ((L + page - 1) // page) * page + k * ((ell + page - 1) // page) * page
    if scheme == "adaptive":
        name = choose_adaptive_tail(k, r, ell, page)
        return allocated_slots(L=L, k=k, ell=ell, page=page, scheme="freeze" if name == "freeze" else "pack")
    raise ValueError(f"unknown scheme {scheme}")


@dataclass(slots=True)
class NodeSnap:
    groups: tuple[list[KVCacheBlock], ...]
    num_tokens: int
    block_size: int
    token_prefix: tuple[int, ...]
    request_id: str = ""
    pinned_blocks: tuple[KVCacheBlock, ...] = ()
    speculative: bool = False


@dataclass
class ForkServeStats:
    fork_aliases: int = 0
    aliased_blocks: int = 0
    snapshots: int = 0
    hash_bypasses: int = 0


@dataclass
class ForkServeTracker:
    """Per-manager tree of CoW snapshots. No-op unless extra_args name a node."""

    snaps: dict[str, NodeSnap] = field(default_factory=dict)
    stats: ForkServeStats = field(default_factory=ForkServeStats)

    def try_alias(
        self, manager: KVCacheManager, request: Request
    ) -> tuple[KVCacheBlocks, int, int] | None:
        extra = extra_of(request)
        parent = extra.get("forkserve_parent")
        if parent is None:
            return None
        key = node_key(extra.get("forkserve_session"), parent)
        snap = self.snaps.get(key) if key is not None else None
        groups_src: list[list[KVCacheBlock]] = []
        parent_tokens = 0
        block_size = 16
        if snap is not None and any(snap.groups):
            prefix = snap.token_prefix
            child_ids = request.all_token_ids
            if prefix and tuple(child_ids[: len(prefix)]) != prefix:
                return None
            groups_src = [list(g) for g in snap.groups]
            parent_tokens = snap.num_tokens
            block_size = snap.block_size
        elif snap is not None and snap.request_id:
            for stm in manager.coordinator.single_type_managers:
                blocks = list(stm.req_to_blocks.get(snap.request_id, ()))
                if not blocks:
                    return None
                groups_src.append(blocks)
                block_size = int(getattr(stm, "block_size", block_size))
            parent_tokens = snap.num_tokens or max(
                (len(g) * block_size for g in groups_src), default=0
            )
        else:
            return None

        groups: list[list[KVCacheBlock]] = []
        n_tokens = 0
        for raw in groups_src:
            full, nt = select_full_blocks(raw, block_size, parent_tokens)
            if not full:
                return None
            groups.append(full)
            n_tokens = max(n_tokens, nt)
        n_tokens = min(n_tokens, max(int(request.num_tokens) - 1, 0))
        n_tokens = (n_tokens // block_size) * block_size
        if n_tokens <= 0:
            return None
        for g in groups:
            manager.block_pool.pin_readonly(g)
        self.stats.fork_aliases += 1
        self.stats.hash_bypasses += 1
        self.stats.aliased_blocks += sum(len(g) for g in groups)
        return manager.create_kv_cache_blocks(tuple(groups)), n_tokens, 0

    def snapshot_if_needed(self, manager: KVCacheManager, request: Request) -> None:
        extra = extra_of(request)
        node = extra.get("forkserve_node")
        if node is None:
            return
        session = extra.get("forkserve_session")
        key = node_key(session, node)
        if key is None:
            return
        groups: list[list[KVCacheBlock]] = []
        block_size = 16
        for stm in manager.coordinator.single_type_managers:
            groups.append(list(stm.req_to_blocks.get(request.request_id, ())))
            block_size = int(getattr(stm, "block_size", block_size))
        num_tokens = int(
            getattr(request, "num_prompt_tokens", 0) or getattr(request, "num_tokens", 0)
        )
        prefix = tuple(request.all_token_ids[:num_tokens])
        prev = self.snaps.get(key)
        already_pinned = bool(prev.pinned_blocks) if prev is not None else False
        snap = NodeSnap(
            groups=tuple(groups),
            num_tokens=num_tokens,
            block_size=block_size,
            token_prefix=prefix,
            request_id=request.request_id,
            pinned_blocks=prev.pinned_blocks if prev is not None else (),
            speculative=extra.get("forkserve_class") == "speculative",
        )
        residual_start = num_tokens // block_size
        if snap.speculative:
            for g in groups:
                for blk in g[residual_start:]:
                    blk.is_speculative = True
        self.snaps[key] = snap
        if already_pinned:
            return
        # Extra-pin only frames this node owns. Inherited RO trunk pages are
        # already pinned by the parent snapshot; touching them again would leak.
        owned: list[KVCacheBlock] = []
        for g in groups:
            for blk in g:
                if blk is None or blk.is_null:
                    continue
                if extra.get("forkserve_parent") is not None and blk.ro:
                    continue
                owned.append(blk)
        if not owned:
            self.stats.snapshots += 1
            return
        manager.block_pool.touch(owned)
        manager.block_pool.pin_readonly(owned)
        r = num_tokens % block_size
        if r:
            last_idx = num_tokens // block_size
            for g in groups:
                if last_idx < len(g):
                    g[last_idx].n_valid = r
                    g[last_idx].ro = True
        snap.pinned_blocks = tuple(owned)
        self.stats.snapshots += 1

    def release(self, manager: KVCacheManager, request: Request) -> None:
        extra = extra_of(request)
        node = extra.get("forkserve_node")
        key = node_key(extra.get("forkserve_session"), node)
        if key is None or key not in self.snaps:
            return
        snap = self.snaps.pop(key)
        if snap.pinned_blocks:
            manager.block_pool.free_blocks(snap.pinned_blocks)
