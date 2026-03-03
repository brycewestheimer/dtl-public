#!/usr/bin/env python3
# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
DTL Python API Benchmarks

Measures performance of core Python binding operations including context
management, container creation, NumPy access, collectives, and algorithms.

Run single-rank:
    python3 bench_python_api.py

Run multi-rank:
    mpirun -np 4 python3 bench_python_api.py
"""

import sys
import time
import statistics

import numpy as np

try:
    import dtl
except ImportError:
    print("ERROR: dtl module not available. Skipping benchmarks.", file=sys.stderr)
    sys.exit(0)


# =============================================================================
# Benchmark Infrastructure
# =============================================================================


def bench(name: str, func, iterations: int = 100, warmup: int = 5):
    """Run a benchmark and print results.

    Args:
        name: Benchmark name
        func: Callable to benchmark
        iterations: Number of timed iterations
        warmup: Number of warmup iterations (not timed)
    """
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = statistics.mean(times)
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)

    print(f"  {name}: avg={avg*1e6:.1f}us  median={med*1e6:.1f}us  "
          f"min={mn*1e6:.1f}us  max={mx*1e6:.1f}us  (n={iterations})")


def section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# =============================================================================
# Context Benchmarks
# =============================================================================


def bench_context():
    """Benchmark context creation and destruction."""
    section("Context Creation / Destruction")

    def create_destroy():
        ctx = dtl.Context()
        del ctx

    bench("Context create+destroy", create_destroy, iterations=50)

    def context_manager():
        with dtl.Context() as ctx:
            pass

    bench("Context manager (with)", context_manager, iterations=50)

    with dtl.Context() as ctx:
        bench("barrier", ctx.barrier, iterations=200)

        def query_props():
            _ = ctx.rank
            _ = ctx.size
            _ = ctx.is_root
            _ = ctx.device_id

        bench("property queries (rank/size/is_root/device_id)", query_props,
              iterations=500)


# =============================================================================
# Vector Creation Benchmarks
# =============================================================================


def bench_vector_creation():
    """Benchmark vector creation for various sizes."""
    section("Vector Creation (float64)")

    with dtl.Context() as ctx:
        for size in [100, 1_000, 10_000, 100_000, 1_000_000]:
            def create(sz=size):
                vec = dtl.DistributedVector(ctx, size=sz, dtype=np.float64)
                del vec

            iters = max(10, 200 // max(1, size // 10_000))
            bench(f"create size={size:>10,}", create, iterations=iters)


def bench_vector_creation_with_fill():
    """Benchmark vector creation with fill value."""
    section("Vector Creation with Fill (float64)")

    with dtl.Context() as ctx:
        for size in [1_000, 100_000, 1_000_000]:
            def create_fill(sz=size):
                vec = dtl.DistributedVector(ctx, size=sz, dtype=np.float64,
                                            fill=0.0)
                del vec

            iters = max(10, 100 // max(1, size // 10_000))
            bench(f"create+fill size={size:>10,}", create_fill, iterations=iters)


# =============================================================================
# Local View / NumPy Access Benchmarks
# =============================================================================


def bench_local_view():
    """Benchmark local_view access and NumPy operations."""
    section("Local View (NumPy) Access")

    with dtl.Context() as ctx:
        for size in [1_000, 100_000, 1_000_000]:
            vec = dtl.DistributedVector(ctx, size=size, dtype=np.float64)

            def get_view():
                _ = vec.local_view()

            bench(f"local_view() size={size:>10,}", get_view, iterations=200)

        # Benchmark NumPy operations on local view
        vec = dtl.DistributedVector(ctx, size=100_000, dtype=np.float64, fill=1.0)
        local = vec.local_view()

        def numpy_sum():
            _ = np.sum(local)

        bench("np.sum on local_view (100k)", numpy_sum, iterations=200)

        def numpy_fill():
            local[:] = 42.0

        bench("np.fill local_view (100k)", numpy_fill, iterations=200)

        def numpy_arange_fill():
            local[:] = np.arange(len(local), dtype=np.float64)

        bench("np.arange fill local_view (100k)", numpy_arange_fill,
              iterations=100)


# =============================================================================
# Collective Benchmarks
# =============================================================================


def bench_collectives():
    """Benchmark collective operations."""
    section("Collective Operations")

    with dtl.Context() as ctx:
        # Allreduce
        for size in [1, 100, 10_000, 100_000]:
            data = np.ones(size, dtype=np.float64)

            def do_allreduce(d=data):
                _ = dtl.allreduce(ctx, d, op="sum")

            iters = max(10, 200 // max(1, size // 1_000))
            bench(f"allreduce sum  size={size:>10,}", do_allreduce,
                  iterations=iters)

        # Broadcast
        for size in [1, 1_000, 100_000]:
            data = np.ones(size, dtype=np.float64)

            def do_bcast(d=data):
                _ = dtl.broadcast(ctx, d, root=0)

            iters = max(10, 100 // max(1, size // 1_000))
            bench(f"broadcast      size={size:>10,}", do_bcast,
                  iterations=iters)

        # Reduce
        for size in [1, 1_000, 100_000]:
            data = np.ones(size, dtype=np.float64)

            def do_reduce(d=data):
                _ = dtl.reduce(ctx, d, op="sum", root=0)

            iters = max(10, 100 // max(1, size // 1_000))
            bench(f"reduce sum     size={size:>10,}", do_reduce,
                  iterations=iters)


# =============================================================================
# Algorithm Benchmarks
# =============================================================================


def bench_algorithms():
    """Benchmark DTL algorithm operations."""
    section("Algorithms")

    with dtl.Context() as ctx:
        # Fill
        for size in [1_000, 100_000, 1_000_000]:
            vec = dtl.DistributedVector(ctx, size=size, dtype=np.float64)

            def do_fill():
                dtl.fill_vector(vec, 42.0)

            iters = max(10, 200 // max(1, size // 10_000))
            bench(f"fill_vector size={size:>10,}", do_fill, iterations=iters)

        # Sort
        for size in [1_000, 10_000, 100_000]:
            vec = dtl.DistributedVector(ctx, size=size, dtype=np.float64)

            def do_sort(v=vec, sz=size):
                # Re-fill with reverse data before each sort
                local = v.local_view()
                for i in range(len(local)):
                    local[i] = float(len(local) - i)
                dtl.sort_vector(v)

            iters = max(5, 50 // max(1, size // 10_000))
            bench(f"sort_vector size={size:>10,}", do_sort, iterations=iters)

        # Transform
        vec = dtl.DistributedVector(ctx, size=100_000, dtype=np.float64, fill=1.0)

        def do_transform():
            dtl.transform_vector(vec, lambda x: x * 2.0)

        bench("transform_vector (100k, x*2)", do_transform, iterations=50)

        # Reduce local
        vec = dtl.DistributedVector(ctx, size=100_000, dtype=np.float64, fill=1.0)

        def do_reduce_local():
            _ = dtl.reduce_local_vector(vec, op="sum")

        bench("reduce_local_vector sum (100k)", do_reduce_local, iterations=100)

        # MinMax
        vec = dtl.DistributedVector(ctx, size=100_000, dtype=np.float64)
        local = vec.local_view()
        local[:] = np.random.randn(len(local))

        def do_minmax():
            _ = dtl.minmax_vector(vec)

        bench("minmax_vector (100k)", do_minmax, iterations=100)


# =============================================================================
# Python API Overhead Comparison
# =============================================================================


def bench_overhead():
    """Compare Python API overhead to raw NumPy computation."""
    section("Python API Overhead vs Raw NumPy")

    size = 100_000
    raw_array = np.ones(size, dtype=np.float64)

    def raw_numpy_sum():
        _ = np.sum(raw_array)

    bench("raw NumPy sum (100k)", raw_numpy_sum, iterations=200)

    with dtl.Context() as ctx:
        vec = dtl.DistributedVector(ctx, size=size, dtype=np.float64, fill=1.0)

        def dtl_local_sum():
            local = vec.local_view()
            _ = np.sum(local)

        bench("DTL local_view + NumPy sum (100k)", dtl_local_sum, iterations=200)

        def dtl_reduce_local():
            _ = dtl.reduce_local_vector(vec, op="sum")

        bench("DTL reduce_local_vector sum (100k)", dtl_reduce_local,
              iterations=200)

    # Fill comparison
    def raw_numpy_fill():
        raw_array[:] = 42.0

    bench("raw NumPy fill (100k)", raw_numpy_fill, iterations=200)

    with dtl.Context() as ctx:
        vec = dtl.DistributedVector(ctx, size=size, dtype=np.float64)

        def dtl_fill():
            dtl.fill_vector(vec, 42.0)

        bench("DTL fill_vector (100k)", dtl_fill, iterations=200)

        def dtl_numpy_fill():
            local = vec.local_view()
            local[:] = 42.0

        bench("DTL local_view + NumPy fill (100k)", dtl_numpy_fill,
              iterations=200)


# =============================================================================
# Main
# =============================================================================


def main():
    # Print header
    with dtl.Context() as ctx:
        print(f"DTL Python API Benchmarks")
        print(f"  DTL version: {dtl.__version__}")
        print(f"  MPI rank: {ctx.rank}/{ctx.size}")
        print(f"  has_mpi={dtl.has_mpi()} has_cuda={dtl.has_cuda()}")

        # Only print from rank 0 when running multi-rank
        if ctx.rank != 0:
            # Non-root ranks still participate in collectives
            bench_collectives_non_root(ctx)
            return

    bench_context()
    bench_vector_creation()
    bench_vector_creation_with_fill()
    bench_local_view()
    bench_collectives()
    bench_algorithms()
    bench_overhead()

    print(f"\n{'=' * 60}")
    print("  Benchmarks complete.")
    print(f"{'=' * 60}")


def bench_collectives_non_root(ctx):
    """Non-root ranks participate in collective benchmarks silently."""
    # Must match the collective calls in bench_collectives
    warmup = 5
    for size in [1, 100, 10_000, 100_000]:
        data = np.ones(size, dtype=np.float64)
        iters = max(10, 200 // max(1, size // 1_000))
        for _ in range(warmup + iters):
            _ = dtl.allreduce(ctx, data, op="sum")

    for size in [1, 1_000, 100_000]:
        data = np.ones(size, dtype=np.float64)
        iters = max(10, 100 // max(1, size // 1_000))
        for _ in range(warmup + iters):
            _ = dtl.broadcast(ctx, data, root=0)

    for size in [1, 1_000, 100_000]:
        data = np.ones(size, dtype=np.float64)
        iters = max(10, 100 // max(1, size // 1_000))
        for _ in range(warmup + iters):
            _ = dtl.reduce(ctx, data, op="sum", root=0)


if __name__ == "__main__":
    main()
