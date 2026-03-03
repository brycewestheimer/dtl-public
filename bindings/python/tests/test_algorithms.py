# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings algorithm module.

Tests cover:
- for_each operations
- copy/fill operations
- find/find_if operations
- count/count_if operations
- reduce_local operations
- sort operations
- minmax operations
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestForEach:
    """Tests for for_each operations."""

    def test_for_each_vector_basic(self, context: "dtl.Context") -> None:
        """Test for_each_vector iterates over all elements."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=5.0)

        # Collect values to verify iteration
        values = []
        dtl.for_each_vector(vec, lambda x: values.append(x))

        assert len(values) == vec.local_size
        assert all(v == 5.0 for v in values)

    def test_for_each_vector_with_index(self, context: "dtl.Context") -> None:
        """Test for_each_vector with index parameter."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=1.0)

        # Collect indices
        indices = []
        dtl.for_each_vector(vec, lambda x, i: indices.append(i), with_index=True)

        assert len(indices) == vec.local_size
        assert indices == list(range(vec.local_size))

    def test_for_each_array(self, context: "dtl.Context") -> None:
        """Test for_each_array iterates over all elements."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=20, dtype=np.int32, fill=7)

        values = []
        dtl.for_each_array(arr, lambda x: values.append(x))

        assert len(values) == arr.local_size
        assert all(v == 7 for v in values)


class TestCopyFill:
    """Tests for copy and fill operations."""

    def test_copy_vector(self, context: "dtl.Context") -> None:
        """Test copy_vector copies data correctly."""
        import dtl
        import numpy as np

        src = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=42.0)
        dst = dtl.DistributedVector(context, size=10, dtype=np.float64)

        dtl.copy_vector(src, dst)

        dst_view = dst.local_view()
        assert np.all(dst_view == 42.0)

    def test_copy_array(self, context: "dtl.Context") -> None:
        """Test copy_array copies data correctly."""
        import dtl
        import numpy as np

        src = dtl.DistributedArray(context, size=15, dtype=np.int64, fill=123)
        dst = dtl.DistributedArray(context, size=15, dtype=np.int64)

        dtl.copy_array(src, dst)

        dst_view = dst.local_view()
        assert np.all(dst_view == 123)

    def test_fill_vector(self, context: "dtl.Context") -> None:
        """Test fill_vector fills all elements."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        dtl.fill_vector(vec, 99.0)

        local = vec.local_view()
        assert np.all(local == 99.0)

    def test_fill_array(self, context: "dtl.Context") -> None:
        """Test fill_array fills all elements."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=20, dtype=np.int32)
        dtl.fill_array(arr, -5)

        local = arr.local_view()
        assert np.all(local == -5)


class TestFind:
    """Tests for find operations."""

    def test_find_vector_found(self, context: "dtl.Context") -> None:
        """Test find_vector finds existing value."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        # Fill with sequential values
        for i in range(len(local)):
            local[i] = i * 2

        if len(local) > 2:
            idx = dtl.find_vector(vec, 4)
            assert idx == 2

    def test_find_vector_not_found(self, context: "dtl.Context") -> None:
        """Test find_vector returns None when not found."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32, fill=0)

        idx = dtl.find_vector(vec, 999)
        assert idx is None

    def test_find_if_vector(self, context: "dtl.Context") -> None:
        """Test find_if_vector with predicate."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        # Fill with negative values, one positive
        local[:] = -1.0
        if len(local) > 3:
            local[3] = 42.0

            idx = dtl.find_if_vector(vec, lambda x: x > 0)
            assert idx == 3

    def test_find_array(self, context: "dtl.Context") -> None:
        """Test find_array finds existing value."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=10, dtype=np.float64, fill=1.0)
        local = arr.local_view()
        if len(local) > 5:
            local[5] = 99.0

            idx = dtl.find_array(arr, 99.0)
            assert idx == 5


class TestCount:
    """Tests for count operations."""

    def test_count_vector(self, context: "dtl.Context") -> None:
        """Test count_vector counts occurrences."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32, fill=5)

        # All elements should match
        count = dtl.count_vector(vec, 5)
        assert count == vec.local_size

        # No elements should match
        count = dtl.count_vector(vec, 99)
        assert count == 0

    def test_count_if_vector(self, context: "dtl.Context") -> None:
        """Test count_if_vector with predicate."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        # Fill with values: 0.5, 1.5, 2.5, ...
        for i in range(len(local)):
            local[i] = i + 0.5

        # Count values > 2.0
        count = dtl.count_if_vector(vec, lambda x: x > 2.0)
        expected = sum(1 for i in range(len(local)) if (i + 0.5) > 2.0)
        assert count == expected

    def test_count_array(self, context: "dtl.Context") -> None:
        """Test count_array counts occurrences."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=20, dtype=np.uint8, fill=0)
        local = arr.local_view()
        # Set some elements to 1
        for i in range(0, len(local), 3):
            local[i] = 1

        count = dtl.count_array(arr, 1)
        expected = (len(local) + 2) // 3
        assert count == expected


class TestReduce:
    """Tests for local reduction operations."""

    def test_reduce_local_vector_sum(self, context: "dtl.Context") -> None:
        """Test reduce_local_vector with sum."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=1.0)

        result = dtl.reduce_local_vector(vec, op="sum")
        assert result == float(vec.local_size)

    def test_reduce_local_vector_prod(self, context: "dtl.Context") -> None:
        """Test reduce_local_vector with product."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=5, dtype=np.float64, fill=2.0)

        result = dtl.reduce_local_vector(vec, op="prod")
        assert result == 2.0 ** vec.local_size

    def test_reduce_local_vector_minmax(self, context: "dtl.Context") -> None:
        """Test reduce_local_vector with min/max."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = i * 10 - 50  # -50, -40, -30, ...

        min_result = dtl.reduce_local_vector(vec, op="min")
        max_result = dtl.reduce_local_vector(vec, op="max")

        assert min_result == -50
        assert max_result == (len(local) - 1) * 10 - 50

    def test_reduce_local_array(self, context: "dtl.Context") -> None:
        """Test reduce_local_array."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=8, dtype=np.float32, fill=0.5)

        result = dtl.reduce_local_array(arr, op="sum")
        assert abs(result - arr.local_size * 0.5) < 0.01


class TestSort:
    """Tests for sorting operations."""

    def test_sort_vector_ascending(self, context: "dtl.Context") -> None:
        """Test sort_vector sorts in ascending order."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        # Fill in reverse order
        for i in range(len(local)):
            local[i] = len(local) - i

        dtl.sort_vector(vec)

        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i-1] <= sorted_local[i]

    def test_sort_vector_descending(self, context: "dtl.Context") -> None:
        """Test sort_vector sorts in descending order."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = float(i)

        dtl.sort_vector(vec, reverse=True)

        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i-1] >= sorted_local[i]

    def test_sort_array(self, context: "dtl.Context") -> None:
        """Test sort_array sorts elements."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=15, dtype=np.int64)
        local = arr.local_view()
        for i in range(len(local)):
            local[i] = (len(local) - i) * 100

        dtl.sort_array(arr)

        sorted_local = arr.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i-1] <= sorted_local[i]


class TestMinMax:
    """Tests for minmax operations."""

    def test_minmax_vector(self, context: "dtl.Context") -> None:
        """Test minmax_vector finds min and max."""
        import dtl
        import numpy as np

        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = i * 10 - 25  # -25, -15, -5, 5, 15, ...

        min_val, max_val = dtl.minmax_vector(vec)

        assert min_val == -25.0
        assert max_val == (len(local) - 1) * 10 - 25.0

    def test_minmax_array(self, context: "dtl.Context") -> None:
        """Test minmax_array finds min and max."""
        import dtl
        import numpy as np

        arr = dtl.DistributedArray(context, size=20, dtype=np.int32)
        local = arr.local_view()
        for i in range(len(local)):
            local[i] = i * i - 100  # i^2 - 100

        min_val, max_val = dtl.minmax_array(arr)

        # Calculate expected
        expected_min = min(i * i - 100 for i in range(len(local)))
        expected_max = max(i * i - 100 for i in range(len(local)))

        assert min_val == expected_min
        assert max_val == expected_max
