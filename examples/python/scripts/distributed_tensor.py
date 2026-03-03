#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
distributed_tensor.py - Distributed tensor with NumPy integration

Demonstrates:
- dtl.DistributedTensor for multi-dimensional distributed data
- NumPy zero-copy local views
- Collective operations on tensor data
- Frobenius norm computation across ranks

Run:
    mpirun -np 4 python distributed_tensor.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Distributed Tensor (Python)")
            print("=================================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # Create a 2D tensor (100 rows x 64 cols)
        nrows, ncols = 100, 64
        tensor = dtl.DistributedTensor(ctx, shape=(nrows, ncols), dtype=np.float64)

        print(f"Rank {ctx.rank}:")
        print(f"  Global shape: {tensor.shape}")
        print(f"  Local shape:  {tensor.local_shape}")
        print(f"  Local size:   {tensor.local_size}")
        ctx.barrier()

        # Get local view as NumPy array
        local = tensor.local_view()

        # Fill: each element = row * 100 + col
        # Local view is flattened; compute row/col from global indices
        local_elements = tensor.local_size
        global_offset = ctx.rank * (nrows // ctx.size) * ncols  # approximate
        for i in range(local_elements):
            global_idx = global_offset + i
            row = global_idx // ncols
            col = global_idx % ncols
            local[i] = float(row * 100 + col)

        ctx.barrier()

        # Compute Frobenius norm (sqrt of sum of squares)
        local_sum_sq = np.sum(local ** 2)
        global_sum_sq = dtl.allreduce(ctx, np.array([local_sum_sq]), op="sum")
        frobenius_norm = np.sqrt(global_sum_sq[0])

        if ctx.is_root:
            # Compute expected norm
            expected_sq = 0.0
            for r in range(nrows):
                for c in range(ncols):
                    val = r * 100 + c
                    expected_sq += val * val
            expected_norm = np.sqrt(expected_sq)

            print(f"\nTensor shape: {nrows} x {ncols}")
            print(f"Total elements: {nrows * ncols}")
            print(f"Frobenius norm: {frobenius_norm:.4f}")
            print(f"Expected norm:  {expected_norm:.4f}")

            ok = abs(frobenius_norm - expected_norm) < 1e-4
            print(f"{'SUCCESS' if ok else 'FAILURE'}")
        ctx.barrier()

        # Demonstrate per-rank statistics
        if ctx.is_root:
            print("\nPer-rank statistics:")
        ctx.barrier()

        local_min = np.min(local)
        local_max = np.max(local)
        local_mean = np.mean(local)
        print(f"  Rank {ctx.rank}: min={local_min:.0f}, max={local_max:.0f}, "
              f"mean={local_mean:.2f}")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
