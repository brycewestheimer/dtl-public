#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
collective_ops.py - Collective communication operations

Demonstrates:
- dtl.allreduce for global reduction
- dtl.broadcast for root-to-all distribution
- dtl.gather for collecting data at root
- dtl.scatter for distributing data from root
- dtl.allgather for all-to-all gathering

Run:
    mpirun -np 4 python collective_ops.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Collective Operations (Python)")
            print("====================================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # 1. Broadcast: root sends 42 to all
        if ctx.is_root:
            print("1. Broadcast:")

        data = np.array([42], dtype=np.int32) if ctx.is_root else np.zeros(1, dtype=np.int32)
        result = dtl.broadcast(ctx, data, root=0)
        print(f"  Rank {ctx.rank}: received {result[0]}")
        ctx.barrier()

        # 2. Allreduce: sum of rank values
        if ctx.is_root:
            print("\n2. Allreduce (sum):")

        local_val = np.array([ctx.rank + 1], dtype=np.float64)
        global_sum = dtl.allreduce(ctx, local_val, op="sum")
        expected = ctx.size * (ctx.size + 1) / 2
        print(f"  Rank {ctx.rank}: local={local_val[0]:.0f}, global_sum={global_sum[0]:.0f}")
        if ctx.is_root:
            print(f"  Expected: {expected:.0f} -> {'OK' if global_sum[0] == expected else 'FAIL'}")
        ctx.barrier()

        # 3. Reduce (to root only)
        if ctx.is_root:
            print("\n3. Reduce (to root):")

        local_val = np.array([ctx.rank * 10.0], dtype=np.float64)
        root_result = dtl.reduce(ctx, local_val, op="sum", root=0)
        if ctx.is_root:
            expected = sum(r * 10.0 for r in range(ctx.size))
            print(f"  Root received: {root_result[0]:.0f} (expected: {expected:.0f})")
        ctx.barrier()

        # 4. Gather: each rank sends rank*10 to root
        if ctx.is_root:
            print("\n4. Gather:")

        send_data = np.array([ctx.rank * 10], dtype=np.int32)
        gathered = dtl.gather(ctx, send_data, root=0)
        if ctx.is_root:
            print(f"  Root gathered: {gathered.flatten().tolist()}")
        ctx.barrier()

        # 5. Scatter: root distributes values
        if ctx.is_root:
            print("\n5. Scatter:")

        scatter_data = None
        if ctx.is_root:
            scatter_data = np.array([(i + 1) * 100 for i in range(ctx.size)], dtype=np.int32)
            print(f"  Root scattering: {scatter_data.tolist()}")

        received = dtl.scatter(ctx, scatter_data, root=0)
        print(f"  Rank {ctx.rank} received: {received.flatten().tolist()}")
        ctx.barrier()

        # 6. Allgather: each rank shares with all
        if ctx.is_root:
            print("\n6. Allgather:")

        my_val = np.array([ctx.rank], dtype=np.int32)
        all_vals = dtl.allgather(ctx, my_val)
        print(f"  Rank {ctx.rank}: {all_vals.flatten().tolist()}")
        ctx.barrier()

        # 7. Min/Max reduction
        if ctx.is_root:
            print("\n7. Min/Max allreduce:")

        local_val = np.array([float(ctx.rank)], dtype=np.float64)
        global_min = dtl.allreduce(ctx, local_val, op="min")
        global_max = dtl.allreduce(ctx, local_val, op="max")
        print(f"  Rank {ctx.rank}: min={global_min[0]:.0f}, max={global_max[0]:.0f}")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
