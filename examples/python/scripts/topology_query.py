#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
topology_query.py - Hardware topology queries

Demonstrates:
- dtl.Topology.num_cpus() for CPU count
- dtl.Topology.num_gpus() for GPU count
- dtl.Topology.is_local() for co-location checking
- dtl.Topology.node_id() for node identification

Run:
    mpirun -np 2 python topology_query.py
"""

import dtl


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Topology Query (Python)")
            print("============================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # Query hardware
        num_cpus = dtl.Topology.num_cpus()
        num_gpus = dtl.Topology.num_gpus()

        print(f"Rank {ctx.rank}:")
        print(f"  CPUs: {num_cpus}")
        print(f"  GPUs: {num_gpus}")

        # CPU affinity
        cpu_id = dtl.Topology.cpu_affinity(ctx.rank)
        print(f"  CPU affinity: {cpu_id}")

        # GPU ID (if available)
        if num_gpus > 0:
            gpu_id = dtl.Topology.gpu_id(ctx.rank)
            print(f"  GPU ID: {gpu_id}")

        # Node ID
        node_id = dtl.Topology.node_id(ctx.rank)
        print(f"  Node ID: {node_id}")

        ctx.barrier()

        # Locality checks
        if ctx.is_root and ctx.size > 1:
            print("\nLocality checks:")
            for r in range(1, ctx.size):
                is_local = dtl.Topology.is_local(0, r)
                status = "same node" if is_local else "different nodes"
                print(f"  Rank 0 & Rank {r}: {status}")

        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
