#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
futures_remote.py - Futures and Remote/RPC operations

Demonstrates:
- dtl.Future for asynchronous value resolution
- dtl.when_all for synchronizing multiple futures
- dtl.Action for action registration
- dtl.remote_invoke for synchronous RPC

Note: Futures and RPC are experimental features. The progress engine
      may have stability issues.

Run:
    mpirun -np 4 python futures_remote.py
"""

import dtl
import numpy as np


def main() -> None:
    with dtl.Context() as ctx:
        if ctx.is_root:
            print("DTL Futures & Remote/RPC (Python)")
            print("===================================")
            print(f"Running with {ctx.size} ranks\n")
        ctx.barrier()

        # --- 1. Basic future: create, set, get ---
        if ctx.is_root:
            print("1. Basic Future:")
        ctx.barrier()

        fut = dtl.Future()
        value = (ctx.rank + 1) * 100
        fut.set(np.array([value], dtype=np.int32).tobytes())
        fut.wait()

        result_bytes = fut.get(4)  # 4 bytes for int32
        result = np.frombuffer(result_bytes, dtype=np.int32)[0]
        print(f"  Rank {ctx.rank}: future value = {result}")
        ctx.barrier()

        # --- 2. Multiple futures ---
        if ctx.is_root:
            print("\n2. Multiple futures:")
        ctx.barrier()

        N = 3
        futures = []
        for i in range(N):
            f = dtl.Future()
            val = ctx.rank * 10 + i
            f.set(np.array([val], dtype=np.int32).tobytes())
            futures.append(f)

        # Wait for all
        try:
            dtl.when_all(futures)
            print(f"  Rank {ctx.rank}: when_all completed")
        except Exception:
            # Fallback: wait individually
            for f in futures:
                f.wait()
            print(f"  Rank {ctx.rank}: waited individually")

        total = 0
        for f in futures:
            val = np.frombuffer(f.get(4), dtype=np.int32)[0]
            total += val

        expected = ctx.rank * 10 * N + sum(range(N))
        print(f"  Rank {ctx.rank}: sum = {total} "
              f"(expected {expected}) {'OK' if total == expected else 'FAIL'}")
        ctx.barrier()

        # --- 3. Remote/RPC ---
        if ctx.is_root:
            print("\n3. Remote/RPC:")
        ctx.barrier()

        # Register action on all ranks
        def square_action(args_bytes):
            val = np.frombuffer(args_bytes, dtype=np.int32)[0]
            result = val * val
            return np.array([result], dtype=np.int32).tobytes()

        try:
            action = dtl.Action(ctx, "square", square_action)
            ctx.barrier()

            # Rank 0 invokes on rank 1
            if ctx.rank == 0 and ctx.size > 1:
                arg_bytes = np.array([7], dtype=np.int32).tobytes()
                result_bytes = dtl.remote_invoke(ctx, target=1,
                                                  action_name="square",
                                                  args=arg_bytes,
                                                  result_size=4)
                if result_bytes:
                    result = np.frombuffer(result_bytes, dtype=np.int32)[0]
                    print(f"  Rank 0: remote_invoke('square', 7) on rank 1 = {result}"
                          f" (expected 49)")
        except Exception as e:
            print(f"  Rank {ctx.rank}: RPC not supported: {e}")
        ctx.barrier()

        if ctx.is_root:
            print("\nDone!")


if __name__ == "__main__":
    main()
