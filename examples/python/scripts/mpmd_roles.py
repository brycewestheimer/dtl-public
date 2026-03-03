#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
mpmd_roles.py - MPMD Role Manager

Demonstrates:
- dtl.RoleManager for role assignment
- dtl.intergroup_send / dtl.intergroup_recv for inter-role communication
- Coordinator/worker pattern

Run:
    mpirun -np 4 python mpmd_roles.py
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
            print("DTL MPMD Role Manager (Python)")
            print("================================")
            print(f"Running with {ctx.size} ranks")
            print(f"Coordinator: rank 0")
            print(f"Workers: ranks 1..{ctx.size - 1}\n")
        ctx.barrier()

        # Create role manager
        mgr = dtl.RoleManager(ctx)

        # Define roles
        mgr.add_role("coordinator", 1)
        mgr.add_role("worker", ctx.size - 1)

        # Initialize (collective)
        mgr.initialize()

        is_coordinator = mgr.has_role("coordinator")
        is_worker = mgr.has_role("worker")

        print(f"Rank {ctx.rank}: coordinator={is_coordinator}, worker={is_worker}")
        ctx.barrier()

        if is_coordinator:
            num_workers = mgr.role_size("worker")
            print(f"\nCoordinator: sending tasks to {num_workers} workers")

            # Send tasks
            for w in range(num_workers):
                task = np.array([(w + 1) * 10], dtype=np.int32)
                dtl.intergroup_send(mgr, "worker", w, task, tag=0)
                print(f"  Sent task {task[0]} to worker {w}")

            # Receive results
            print("\nResults:")
            for w in range(num_workers):
                result = dtl.intergroup_recv(mgr, "worker", w,
                                              count=1, dtype=np.int32, tag=1)
                print(f"  Worker {w} returned: {result[0]}")

        if is_worker:
            # Receive task
            task = dtl.intergroup_recv(mgr, "coordinator", 0,
                                        count=1, dtype=np.int32, tag=0)
            worker_rank = mgr.role_rank("worker")

            # Compute: square the value
            result = np.array([task[0] ** 2], dtype=np.int32)
            print(f"Worker {worker_rank} (global rank {ctx.rank}): "
                  f"received {task[0]}, computed {result[0]}")

            # Send result
            dtl.intergroup_send(mgr, "coordinator", 0, result, tag=1)

        ctx.barrier()
        mgr.destroy()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
