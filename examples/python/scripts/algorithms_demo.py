#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
algorithms_demo.py - Algorithm operations

Demonstrates:
- dtl.transform_vector for element-wise transformation
- dtl.sort_vector for local sorting
- dtl.inclusive_scan_vector / dtl.exclusive_scan_vector for prefix sums
- dtl.find_vector / dtl.count_if_vector for search and counting
- dtl.minmax_vector for min/max
- dtl.reduce_local_vector for local reduction

Run:
    mpirun -np 4 python algorithms_demo.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Algorithm Operations (Python)")
            print("===================================\n")
        ctx.barrier()

        # Create a vector and initialize
        vec = dtl.DistributedVector(ctx, size=20, dtype=np.float64)
        local = vec.local_view()
        offset = vec.local_offset
        local[:] = np.arange(offset + 1, offset + 1 + len(local), dtype=np.float64)

        print(f"Rank {ctx.rank}: initial values = {local[:min(8, len(local))].tolist()}")
        ctx.barrier()

        # 1. Transform: multiply by 2
        if ctx.is_root:
            print("\n1. Transform (x * 2):")
        ctx.barrier()

        dtl.transform_vector(vec, lambda x: x * 2)
        print(f"  Rank {ctx.rank}: {local[:min(8, len(local))].tolist()}")
        ctx.barrier()

        # 2. Local reduction: sum
        if ctx.is_root:
            print("\n2. Local reduction (sum):")
        ctx.barrier()

        local_sum = dtl.reduce_local_vector(vec, op="sum")
        print(f"  Rank {ctx.rank}: local sum = {local_sum}")

        # Global reduction
        global_sum = dtl.allreduce(ctx, np.array([local_sum]), op="sum")
        if ctx.is_root:
            print(f"  Global sum = {global_sum[0]}")
        ctx.barrier()

        # 3. Min/Max
        if ctx.is_root:
            print("\n3. Min/Max:")
        ctx.barrier()

        min_val, max_val = dtl.minmax_vector(vec)
        print(f"  Rank {ctx.rank}: min={min_val}, max={max_val}")
        ctx.barrier()

        # 4. Find
        if ctx.is_root:
            print("\n4. Find (value 10.0):")
        ctx.barrier()

        idx = dtl.find_vector(vec, 10.0)
        if idx is not None:
            print(f"  Rank {ctx.rank}: found at local index {idx}")
        else:
            print(f"  Rank {ctx.rank}: not found locally")
        ctx.barrier()

        # 5. Count with predicate
        if ctx.is_root:
            print("\n5. Count (elements > 20):")
        ctx.barrier()

        count = dtl.count_if_vector(vec, lambda x: x > 20)
        print(f"  Rank {ctx.rank}: {count} elements > 20")
        ctx.barrier()

        # 6. Sort
        if ctx.is_root:
            print("\n6. Sort (descending):")
        ctx.barrier()

        dtl.sort_vector(vec, reverse=True)
        print(f"  Rank {ctx.rank}: {local[:min(8, len(local))].tolist()}")
        ctx.barrier()

        # 7. Inclusive scan
        if ctx.is_root:
            print("\n7. Inclusive scan (sum):")
        ctx.barrier()

        # Reset to 1s for scan demonstration
        scan_vec = dtl.DistributedVector(ctx, size=12, dtype=np.float64, fill=1.0)
        dtl.inclusive_scan_vector(scan_vec, op="sum")
        scan_local = scan_vec.local_view()
        print(f"  Rank {ctx.rank}: {scan_local.tolist()}")
        ctx.barrier()

        # 8. Exclusive scan
        if ctx.is_root:
            print("\n8. Exclusive scan (sum):")
        ctx.barrier()

        exscan_vec = dtl.DistributedVector(ctx, size=12, dtype=np.float64, fill=1.0)
        dtl.exclusive_scan_vector(exscan_vec, op="sum")
        exscan_local = exscan_vec.local_view()
        print(f"  Rank {ctx.rank}: {exscan_local.tolist()}")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
