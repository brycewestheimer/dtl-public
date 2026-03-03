#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
point_to_point.py - Point-to-point communication

Demonstrates:
- dtl.send / dtl.recv for blocking P2P
- dtl.sendrecv for combined send/receive
- Ring communication pattern

Run:
    mpirun -np 4 python point_to_point.py
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
            print("DTL Point-to-Point Communication (Python)")
            print("===========================================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # 1. Simple send/recv between rank 0 and rank 1
        if ctx.is_root:
            print("1. Send/Recv (rank 0 -> rank 1):")
        ctx.barrier()

        if ctx.rank == 0:
            data = np.array([42], dtype=np.int32)
            dtl.send(ctx, data, dest=1, tag=0)
            print(f"  Rank 0 sent: {data[0]}")
        elif ctx.rank == 1:
            received = dtl.recv(ctx, count=1, dtype=np.int32, source=0, tag=0)
            print(f"  Rank 1 received: {received[0]}")
        ctx.barrier()

        # 2. Sendrecv ring: each rank exchanges with next/prev
        if ctx.is_root:
            print("\n2. Sendrecv Ring:")
        ctx.barrier()

        next_rank = (ctx.rank + 1) % ctx.size
        prev_rank = (ctx.rank + ctx.size - 1) % ctx.size

        send_data = np.array([ctx.rank * 100], dtype=np.int32)
        received = dtl.sendrecv(ctx, send_data, dest=next_rank,
                                recvcount=1, source=prev_rank,
                                sendtag=10, recvtag=10)

        expected = prev_rank * 100
        print(f"  Rank {ctx.rank}: sent {send_data[0]} to rank {next_rank}, "
              f"received {received[0]} from rank {prev_rank} "
              f"({'OK' if received[0] == expected else 'FAIL'})")
        ctx.barrier()

        # 3. Multiple exchanges: pipeline pattern
        if ctx.is_root:
            print("\n3. Pipeline (multi-stage sendrecv):")
        ctx.barrier()

        value = np.array([float(ctx.rank)], dtype=np.float64)

        # Pass values around the ring 3 times
        for stage in range(3):
            value = dtl.sendrecv(ctx, value, dest=next_rank,
                                 recvcount=1, recvdtype=np.float64,
                                 source=prev_rank,
                                 sendtag=20 + stage, recvtag=20 + stage)

        # After 3 rotations, each rank should have (rank - 3) mod size
        expected_origin = (ctx.rank - 3) % ctx.size
        print(f"  Rank {ctx.rank}: value = {value[0]:.0f} "
              f"(expected: {expected_origin})")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
