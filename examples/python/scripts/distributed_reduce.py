#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
distributed_reduce.py - Distributed reduction example

This script demonstrates:
- Creating distributed vectors
- Computing local operations
- Distributed reduction using MPI

Run with MPI:
    mpirun -np 4 python distributed_reduce.py
"""

import dtl
import numpy as np


def main() -> None:
    """Main entry point."""
    # Create context
    with dtl.Context() as ctx:
        # Problem size
        global_size = 10000

        # Create distributed vector
        vec = dtl.DistributedVector(ctx, size=global_size, dtype=np.float64)

        # Get local view and initialize
        local = vec.local_view()

        # Each rank fills its portion with values based on global index
        start_idx = vec.local_offset
        local[:] = np.arange(start_idx, start_idx + len(local), dtype=np.float64)

        # Compute local sum
        local_sum = np.sum(local)
        print(f"Rank {ctx.rank}: local_sum = {local_sum:.2f} "
              f"(indices {start_idx} to {start_idx + len(local) - 1})")

        # Synchronize before reduction
        ctx.barrier()

        # For distributed reduction, we need mpi4py
        try:
            from mpi4py import MPI

            # Perform allreduce
            global_sum = MPI.COMM_WORLD.allreduce(local_sum, op=MPI.SUM)

            # Expected sum: 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
            expected = global_size * (global_size - 1) / 2

            if ctx.is_root:
                print(f"\nGlobal sum: {global_sum:.2f}")
                print(f"Expected:   {expected:.2f}")
                print(f"Match: {np.isclose(global_sum, expected)}")

        except ImportError:
            # Without mpi4py, just print local results
            if ctx.is_root:
                print("\nmpi4py not available - showing local sum only")
                expected_local = sum(range(vec.local_size))
                print(f"Expected local sum (rank 0): {expected_local:.2f}")

        # Final barrier
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
