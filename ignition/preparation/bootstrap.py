# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared preparation configuration fields and plan-forwarding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ignition.preparation.tuning_replicas import TuningReplica


def validate_host_topology(roce_ring: bool, roce_chain: bool, hosts: tuple[str, ...] | None) -> None:
    """Raise ValueError when requested RoCE ring/chain flags disagree with the host layout."""
    from ignition.planning import host_shape_error

    error = host_shape_error(roce_ring, roce_chain, None if hosts is None else len(hosts))
    if error is not None:
        raise ValueError(error)


@dataclass(frozen=True)
class BasePreparationConfig:
    """Common preparation settings accepted by every model recipe bootstrap."""

    compile_workers: int | None = None
    autotune_effort: str = "default"
    native_dense_activations: bool = False
    lock_gpu_clocks: bool = False
    tuning_replica: TuningReplica | None = None
    context: int = 262144
    prefill_chunk_size: int = 4096
    max_active: int = 4
    snapshot_budget_bytes: int = 4 << 30
    headroom_bytes: int = 4 << 30
    compile_decoder: bool = True
    retain_graphs: bool = False
    kernel_cache: Path | None = None
    mtp_only: bool = False
    speculative_widths: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speculative_fixed_width: int | None = None
    speculative_token_capacity: int | None = None


def common_preparation_kwargs(plan, *, compile_forwards: bool, kernel_cache: Path | None,
                              retain_graphs: bool) -> dict:
    """Extract the shared PreparationConfig keyword arguments from a sizing Plan."""
    return {
        "mtp_only": plan.speculation is not None and plan.speculation.mtp_only,
        "compile_workers": plan.compile_workers,
        "autotune_effort": plan.autotune_effort,
        "tuning_replica": plan.tuning_replica,
        "native_dense_activations": plan.native_dense_activations,
        "lock_gpu_clocks": plan.lock_gpu_clocks and plan.autotune_effort != "off",
        "max_active": plan.concurrency,
        "context": plan.declared_context,
        "prefill_chunk_size": plan.prefill_chunk,
        "snapshot_budget_bytes": plan.snapshots * plan.costs.state_row_bytes,
        "headroom_bytes": plan.headroom_bytes + plan.reserve_bytes,
        "compile_decoder": compile_forwards,
        "retain_graphs": retain_graphs,
        "kernel_cache": kernel_cache,
    }
