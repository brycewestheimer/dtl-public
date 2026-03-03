#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
monte_carlo_pi.py - Monte Carlo Pi estimation

Demonstrates:
- dtl.DistributedVector for local storage
- NumPy random sampling
- dtl.allreduce for global aggregation
- Parallel random number generation with rank-specific seeds

Run:
    mpirun -np 4 python monte_carlo_pi.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Monte Carlo Pi Estimation (Python)")
            print("========================================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # Each rank samples N points
        samples_per_rank = 1_000_000
        total_samples = samples_per_rank * ctx.size

        # Rank-specific seed for independent random streams
        rng = np.random.default_rng(seed=42 + ctx.rank * 12345)

        # Generate random (x, y) points
        x = rng.random(samples_per_rank)
        y = rng.random(samples_per_rank)

        # Count points inside unit circle
        local_hits = int(np.sum(x * x + y * y <= 1.0))
        local_pi = 4.0 * local_hits / samples_per_rank

        print(f"  Rank {ctx.rank}: {local_hits:,} / {samples_per_rank:,} hits "
              f"(local pi ~ {local_pi:.6f})")
        ctx.barrier()

        # Global reduction
        local_array = np.array([local_hits], dtype=np.int64)
        global_hits_array = dtl.allreduce(ctx, local_array, op="sum")
        global_hits = int(global_hits_array[0])

        pi_estimate = 4.0 * global_hits / total_samples
        error = abs(pi_estimate - np.pi)

        if ctx.is_root:
            print(f"\nTotal samples: {total_samples:,}")
            print(f"Total hits:    {global_hits:,}")
            print(f"Pi estimate:   {pi_estimate:.8f}")
            print(f"Actual pi:     {np.pi:.8f}")
            print(f"Error:         {error:.4e}")
            status = "SUCCESS" if error < 0.01 else "WARNING"
            within = "within" if error < 0.01 else "outside"
            print(f"{status}: Estimate {within} 0.01 tolerance")

        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
