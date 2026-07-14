# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, const_expr
from flydsl.expr.typing import T

from .tensor_shim import GTensor


@functools.lru_cache(maxsize=1024)
def compile_grouped_tile_map_kernel(TILE_M: int, GROUP_COUNT: int):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def grouped_tile_map_kernel(
        OFFS: fx.Pointer,
        TILE_GROUPS: fx.Pointer,
        TILE_MS: fx.Pointer,
        TILE_NS: fx.Pointer,
        TOTAL_TILES: fx.Pointer,
        n_blocks: fx.Int32,
        max_m_tiles_upper: fx.Int32,
        total_tiles_upper: fx.Int32,
    ):
        OFFS_ = GTensor(OFFS, dtype=T.i32, shape=(GROUP_COUNT,))
        TILE_GROUPS_ = GTensor(TILE_GROUPS, dtype=T.i32, shape=(-1,))
        TILE_MS_ = GTensor(TILE_MS, dtype=T.i32, shape=(-1,))
        TILE_NS_ = GTensor(TILE_NS, dtype=T.i32, shape=(-1,))
        TOTAL_TILES_ = GTensor(TOTAL_TILES, dtype=T.i32, shape=(1,))

        # The append counter must be initialized by the caller before launch.
        candidate = fx.Int32(fx.block_idx.x * 256 + fx.thread_idx.x)
        in_range = arith.cmpi(
            arith.CmpIPredicate.ult, candidate, total_tiles_upper
        )
        candidate_if = scf.IfOp(in_range, results_=[], has_else=False)
        with ir.InsertionPoint(candidate_if.then_block):
            group_span = max_m_tiles_upper * n_blocks
            group_idx_i32 = candidate // group_span
            tile_in_group = candidate - group_idx_i32 * group_span
            tile_m_idx = tile_in_group // n_blocks
            tile_n_idx = tile_in_group - tile_m_idx * n_blocks
            group_idx = fx.Index(group_idx_i32)
            group_end = OFFS_[group_idx]
            has_prev_group = arith.cmpi(
                arith.CmpIPredicate.ugt, group_idx_i32, fx.Int32(0)
            )
            prev_group = arith.select(has_prev_group, group_idx_i32 - 1, fx.Int32(0))
            group_start = arith.select(
                has_prev_group, OFFS_[fx.Index(prev_group)], fx.Int32(0)
            )
            group_m = group_end - group_start
            group_m_tiles = (group_m + fx.Int32(TILE_M - 1)) // fx.Int32(TILE_M)
            valid_tile = arith.cmpi(
                arith.CmpIPredicate.ult, tile_m_idx, group_m_tiles
            )
            valid_if = scf.IfOp(valid_tile, results_=[], has_else=False)
            with ir.InsertionPoint(valid_if.then_block):
                counter_addr = TOTAL_TILES_.get_llvm_ptr(
                    TOTAL_TILES, arith.constant(0, index=True)
                )
                counter_ptr = llvm.IntToPtrOp(
                    ir.Type.parse("!llvm.ptr<1>"), counter_addr
                ).result
                counter_ptr_v = (
                    counter_ptr._value
                    if const_expr(hasattr(counter_ptr, "_value"))
                    else counter_ptr
                )
                write_idx = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    counter_ptr_v,
                    arith.constant(1, type=T.i32),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                TILE_GROUPS_[fx.Index(write_idx)] = group_idx_i32
                TILE_MS_[fx.Index(write_idx)] = tile_m_idx
                TILE_NS_[fx.Index(write_idx)] = tile_n_idx
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_grouped_tile_map_kernel(
        OFFS: fx.Pointer,
        TILE_GROUPS: fx.Pointer,
        TILE_MS: fx.Pointer,
        TILE_NS: fx.Pointer,
        TOTAL_TILES: fx.Pointer,
        n_blocks: fx.Int32,
        max_m_tiles_upper: fx.Int32,
        total_tiles_upper: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = (total_tiles_upper + 255) // 256
        grouped_tile_map_kernel(
            OFFS,
            TILE_GROUPS,
            TILE_MS,
            TILE_NS,
            TOTAL_TILES,
            n_blocks,
            max_m_tiles_upper,
            total_tiles_upper,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_grouped_tile_map_kernel
