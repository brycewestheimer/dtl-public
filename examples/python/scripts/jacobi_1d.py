#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
jacobi_1d.py - 1D Jacobi iterative solver

Solves u''(x) = 0 with boundary conditions u(0) = 1, u(L) = 0.
Uses Jacobi iteration with halo exchange via dtl.send / dtl.recv.

Demonstrates:
- Halo exchange using dtl.send / dtl.recv
- Convergence checking via dtl.allreduce with op="max"
- NumPy vectorized Jacobi update

Run:
    mpirun -np 4 python jacobi_1d.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL 1D Jacobi Solver (Python)")
            print("==============================")
            print(f"Ranks: {ctx.size}")
        ctx.barrier()

        # Problem setup
        global_n = 100  # Interior points
        max_iter = 10000
        tol = 1e-8

        # Partition interior points among ranks
        local_n = global_n // ctx.size
        remainder = global_n % ctx.size
        if ctx.rank < remainder:
            local_n += 1

        # Arrays with halo cells: [left_halo | interior | right_halo]
        u = np.zeros(local_n + 2, dtype=np.float64)
        u_new = np.zeros(local_n + 2, dtype=np.float64)

        # Boundary conditions: u(0) = 1.0 on leftmost rank
        if ctx.rank == 0:
            u[0] = 1.0
            u_new[0] = 1.0
        # u(L) = 0.0 on rightmost rank (already zero)

        halo_tag = 10

        if ctx.is_root:
            print(f"Grid: {global_n} interior points")
            print(f"BCs: u(0)=1, u(L)=0")
            print(f"Tolerance: {tol:.0e}\n")
        ctx.barrier()

        global_diff = 0.0

        for iteration in range(max_iter):
            # --- Halo exchange ---
            # Send right boundary to right neighbor
            if ctx.rank < ctx.size - 1:
                dtl.send(ctx, np.array([u[local_n]], dtype=np.float64),
                         dest=ctx.rank + 1, tag=halo_tag)
            if ctx.rank > 0:
                left_halo = dtl.recv(ctx, count=1, dtype=np.float64,
                                     source=ctx.rank - 1, tag=halo_tag)
                u[0] = left_halo[0]

            # Send left boundary to left neighbor
            if ctx.rank > 0:
                dtl.send(ctx, np.array([u[1]], dtype=np.float64),
                         dest=ctx.rank - 1, tag=halo_tag + 1)
            if ctx.rank < ctx.size - 1:
                right_halo = dtl.recv(ctx, count=1, dtype=np.float64,
                                      source=ctx.rank + 1, tag=halo_tag + 1)
                u[local_n + 1] = right_halo[0]

            # --- Jacobi update (vectorized) ---
            u_new[1:local_n + 1] = 0.5 * (u[0:local_n] + u[2:local_n + 2])

            # Compute local max diff
            local_diff = float(np.max(np.abs(
                u_new[1:local_n + 1] - u[1:local_n + 1])))

            # Copy new to old
            u[1:local_n + 1] = u_new[1:local_n + 1]

            # Check convergence: global max diff
            diff_arr = np.array([local_diff], dtype=np.float64)
            global_diff_arr = dtl.allreduce(ctx, diff_arr, op="max")
            global_diff = float(global_diff_arr[0])

            if global_diff < tol:
                break

        ctx.barrier()

        if ctx.is_root:
            print(f"Converged after {iteration} iterations")
            print(f"Final max diff: {global_diff:.4e}\n")

        # Print solution samples
        for r in range(ctx.size):
            if ctx.rank == r:
                print(f"  Rank {ctx.rank}: u[first]={u[1]:.6f}, "
                      f"u[last]={u[local_n]:.6f}")
            ctx.barrier()

        if ctx.is_root:
            print("\nExpected: linear from 1.0 to 0.0")
            status = "SUCCESS" if global_diff < tol else "FAILURE"
            state = "converged" if global_diff < tol else "did not converge"
            print(f"{status}: Solver {state}")


if __name__ == "__main__":
    main()
