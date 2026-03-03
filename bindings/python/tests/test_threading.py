# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for GIL release on blocking DTL operations.

These tests verify that the GIL is correctly released during blocking
operations, allowing other Python threads to run concurrently. Full
multi-threaded MPI testing requires launching with mpirun and multiple
ranks; these single-process tests verify that:

1. Container creation and local operations work from the main thread.
2. Collective operations complete without deadlocking due to GIL issues.
3. Threading primitives can be used alongside DTL operations.

Note: True multi-rank multi-threaded testing requires MPI support and
should be run via: mpirun -np 2 python -m pytest test_threading.py
"""

import threading
import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestMainThreadOperations:
    """Verify basic operations work from the main thread with GIL release enabled."""

    def test_create_vector_main_thread(self, context: "dtl.Context") -> None:
        """Test that creating a distributed vector works on the main thread."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        assert vec.global_size == 100
        assert vec.local_size > 0

    def test_create_vector_with_fill(self, context: "dtl.Context") -> None:
        """Test that creating a vector with fill works (no GIL issue)."""
        import dtl

        vec = dtl.DistributedVector(context, size=50, dtype=np.float64, fill=3.14)
        local = vec.local_view()
        assert np.allclose(local, 3.14)

    def test_vector_local_view(self, context: "dtl.Context") -> None:
        """Test that local_view (non-blocking, no GIL release) works correctly."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=1.0)
        local = vec.local_view()
        assert isinstance(local, np.ndarray)
        assert local.dtype == np.float64
        assert np.all(local == 1.0)


class TestCollectiveWithGILRelease:
    """Verify collective operations work with GIL release."""

    def test_allreduce_sum(self, context: "dtl.Context") -> None:
        """Test allreduce with GIL release does not deadlock or corrupt data."""
        import dtl

        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = dtl.allreduce(context, data, op="sum")

        # In single-process mode, allreduce of [1, 2, 3] should return [1, 2, 3]
        assert result.shape == data.shape
        assert result.dtype == data.dtype
        # With 1 rank, the result equals the input
        expected = data * context.size
        assert np.allclose(result, expected)

    def test_broadcast(self, context: "dtl.Context") -> None:
        """Test broadcast with GIL release works correctly."""
        import dtl

        data = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        result = dtl.broadcast(context, data, root=0)

        assert result.shape == data.shape
        assert np.allclose(result, data)

    def test_reduce_sum(self, context: "dtl.Context") -> None:
        """Test reduce with GIL release works correctly."""
        import dtl

        data = np.array([1.0, 2.0], dtype=np.float64)
        result = dtl.reduce(context, data, op="sum", root=0)

        if context.is_root:
            expected = data * context.size
            assert np.allclose(result, expected)

    def test_barrier(self, context: "dtl.Context") -> None:
        """Test barrier with GIL release does not deadlock."""
        context.barrier()
        # If we reach here, barrier completed successfully

    def test_gather(self, context: "dtl.Context") -> None:
        """Test gather with GIL release works correctly."""
        import dtl

        data = np.array([float(context.rank)], dtype=np.float64)
        result = dtl.gather(context, data, root=0)

        if context.is_root:
            assert result.shape[0] == context.size

    def test_scatter(self, context: "dtl.Context") -> None:
        """Test scatter with GIL release works correctly."""
        import dtl

        data = np.arange(context.size, dtype=np.float64).reshape(context.size, 1)
        result = dtl.scatter(context, data, root=0)
        assert result.shape == (1,)


class TestVectorSyncWithGILRelease:
    """Verify vector sync (barrier) works with GIL release."""

    def test_vector_sync(self, context: "dtl.Context") -> None:
        """Test that vector sync (which calls barrier) works with GIL release."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=1.0)
        vec.sync()
        # If we reach here, sync completed without deadlock


class TestThreadSafety:
    """Basic thread safety tests.

    These tests verify that GIL release does not cause crashes when
    Python threads interact with DTL objects. Full multi-threaded
    correctness testing with multiple MPI ranks requires running
    under mpirun with MPI_THREAD_MULTIPLE support.
    """

    def test_concurrent_local_work_during_noop(self, context: "dtl.Context") -> None:
        """Test that a background thread can do Python work while main thread
        calls a blocking DTL operation (barrier in single-process mode is fast,
        but exercises the GIL release path)."""
        import dtl

        results = []

        def background_work() -> None:
            """Do some Python work in a background thread."""
            total = sum(range(1000))
            results.append(total)

        thread = threading.Thread(target=background_work)
        thread.start()

        # Main thread calls barrier (GIL should be released during the call)
        context.barrier()

        thread.join(timeout=5.0)
        assert not thread.is_alive(), "Background thread did not complete"
        assert len(results) == 1
        assert results[0] == sum(range(1000))

    def test_vector_operations_sequential_threads(
        self, context: "dtl.Context"
    ) -> None:
        """Test that vector operations work correctly when called from
        threads sequentially (not concurrently on the same object)."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=0.0)
        errors = []

        def fill_and_check(value: float) -> None:
            """Fill vector and verify (called sequentially)."""
            try:
                vec.fill(value)
                local = vec.local_view()
                assert np.all(local == value)
            except Exception as e:
                errors.append(e)

        # Run sequentially in threads (not concurrent -- tests thread safety
        # of the Python/C++ boundary, not concurrent access)
        for val in [1.0, 2.0, 3.0]:
            thread = threading.Thread(target=fill_and_check, args=(val,))
            thread.start()
            thread.join(timeout=5.0)
            assert not thread.is_alive()

        assert len(errors) == 0, f"Errors in threads: {errors}"
