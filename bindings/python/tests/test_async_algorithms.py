# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings async algorithm operations (Phase 28).

Tests cover:
- AlgorithmFuture wait() and get() semantics
- async_for_each with vectors and arrays
- async_transform with vectors and arrays
- async_reduce with various operations
- async_sort with ascending and descending order
- done() non-blocking completion check
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestAlgorithmFuture:
    """Tests for AlgorithmFuture basics."""

    def test_future_wait_completes(self) -> None:
        """Test that AlgorithmFuture.wait() completes."""
        import dtl

        fut = dtl.AlgorithmFuture(lambda: 42)
        fut.wait()
        assert fut.done() is True

    def test_future_get_returns_result(self) -> None:
        """Test that AlgorithmFuture.get() returns the result."""
        import dtl

        fut = dtl.AlgorithmFuture(lambda: 42)
        result = fut.get()
        assert result == 42

    def test_future_get_raises_on_error(self) -> None:
        """Test that get() re-raises exceptions from the worker."""
        import dtl

        def fail():
            raise ValueError("test error")

        fut = dtl.AlgorithmFuture(fail)
        with pytest.raises(ValueError, match="test error"):
            fut.get()

    def test_future_repr(self) -> None:
        """Test AlgorithmFuture repr."""
        import dtl

        fut = dtl.AlgorithmFuture(lambda: None)
        fut.wait()
        assert "done" in repr(fut)


class TestAsyncForEach:
    """Tests for async_for_each."""

    def test_async_for_each_vector(self, context: "dtl.Context") -> None:
        """Test async_for_each iterates vector elements."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=5.0)
        values = []
        fut = dtl.async_for_each(vec, lambda x: values.append(x))
        fut.wait()

        assert len(values) == vec.local_size
        assert all(v == 5.0 for v in values)

    def test_async_for_each_array(self, context: "dtl.Context") -> None:
        """Test async_for_each iterates array elements."""
        import dtl

        arr = dtl.DistributedArray(context, size=10, dtype=np.int32, fill=7)
        values = []
        fut = dtl.async_for_each(arr, lambda x: values.append(x))
        fut.wait()

        assert len(values) == arr.local_size
        assert all(v == 7 for v in values)


class TestAsyncTransform:
    """Tests for async_transform."""

    def test_async_transform_vector(self, context: "dtl.Context") -> None:
        """Test async_transform doubles vector values."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=3.0)
        fut = dtl.async_transform(vec, lambda x: x * 2)
        fut.wait()

        local = vec.local_view()
        assert np.all(local == 6.0)

    def test_async_transform_array(self, context: "dtl.Context") -> None:
        """Test async_transform squares array values."""
        import dtl

        arr = dtl.DistributedArray(context, size=8, dtype=np.int32, fill=4)
        fut = dtl.async_transform(arr, lambda x: x * x)
        fut.wait()

        local = arr.local_view()
        assert np.all(local == 16)


class TestAsyncReduce:
    """Tests for async_reduce."""

    def test_async_reduce_sum(self, context: "dtl.Context") -> None:
        """Test async_reduce with sum operation."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=1.0)
        fut = dtl.async_reduce(vec, op="sum")
        result = fut.get()

        assert result == float(vec.local_size)

    def test_async_reduce_max(self, context: "dtl.Context") -> None:
        """Test async_reduce with max operation."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = i * 10

        fut = dtl.async_reduce(vec, op="max")
        result = fut.get()

        assert result == (vec.local_size - 1) * 10


class TestAsyncSort:
    """Tests for async_sort."""

    def test_async_sort_vector_ascending(self, context: "dtl.Context") -> None:
        """Test async_sort sorts vector in ascending order."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = len(local) - i

        fut = dtl.async_sort(vec)
        fut.wait()

        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i - 1] <= sorted_local[i]

    def test_async_sort_vector_descending(self, context: "dtl.Context") -> None:
        """Test async_sort sorts vector in descending order."""
        import dtl

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = float(i)

        fut = dtl.async_sort(vec, reverse=True)
        fut.wait()

        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i - 1] >= sorted_local[i]
