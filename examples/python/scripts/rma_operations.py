#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
rma_operations.py - RMA Put/Get operations

Demonstrates:
- dtl.Window for RMA window creation
- dtl.rma_put / dtl.rma_get for one-sided data transfer
- dtl.rma_accumulate for atomic remote accumulation
- Fence synchronization

Note: RMA may fail on some MPI implementations (e.g., WSL2 OpenMPI 4.1.6).

Run:
    mpirun -np 2 python rma_operations.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.size < 2:
            if ctx.is_root:
                print("This example requires at least 2 ranks.")
            return

        if ctx.is_root:
            print("DTL RMA Operations (Python)")
            print("============================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # Create local buffer and expose via RMA window
        local_buf = np.array([ctx.rank * 100, ctx.rank * 100 + 1,
                               ctx.rank * 100 + 2, ctx.rank * 100 + 3],
                              dtype=np.int32)

        try:
            win = dtl.Window(ctx, local_buf)
        except Exception as e:
            print(f"Rank {ctx.rank}: Window creation failed: {e}")
            return

        # Fence to open epoch
        win.fence()

        # 1. Put: rank 0 writes to rank 1
        if ctx.is_root:
            print("1. RMA Put (rank 0 -> rank 1):")

        if ctx.rank == 0:
            data = np.array([999], dtype=np.int32)
            dtl.rma_put(win, target=1, offset=0, data=data)
            print(f"  Rank 0: put value 999 to rank 1")

        win.fence()

        if ctx.rank == 1:
            print(f"  Rank 1: buffer[0] = {local_buf[0]} (expected 999)")
        ctx.barrier()

        # 2. Get: rank 1 reads from rank 0
        if ctx.is_root:
            print("\n2. RMA Get (rank 1 reads rank 0):")

        if ctx.rank == 1:
            result = dtl.rma_get(win, target=0, offset=2 * 4,
                                  size=4, dtype=np.int32)
            print(f"  Rank 1: got value {result[0]} from rank 0 "
                  f"(expected {0 * 100 + 2})")

        win.fence()
        ctx.barrier()

        # 3. Accumulate: all ranks atomically add to rank 0
        if ctx.is_root:
            print("\n3. RMA Accumulate (all -> rank 0):")

        # Reset rank 0's first element
        if ctx.rank == 0:
            local_buf[0] = 0
        win.fence()

        # Each rank accumulates its rank number
        data = np.array([ctx.rank], dtype=np.int32)
        dtl.rma_accumulate(win, target=0, offset=0, data=data, op="sum")

        win.fence()

        if ctx.rank == 0:
            expected = sum(range(ctx.size))
            print(f"  Rank 0: accumulated value = {local_buf[0]} "
                  f"(expected {expected})")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
