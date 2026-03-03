# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings container module.

Tests cover:
- DistributedVector creation and operations
- DistributedArray creation and operations (Phase 10A)
- DistributedTensor creation and operations
- NumPy integration (local_view, zero-copy)
- Multiple dtypes
- Fill operations
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestDistributedVectorCreation:
    """Tests for DistributedVector creation."""

    def test_create_f64_vector(self, context: "dtl.Context") -> None:
        """Test creating float64 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        assert vec.global_size == 100
        assert vec.local_size > 0
        assert vec.local_size <= 100

    def test_create_f32_vector(self, context: "dtl.Context") -> None:
        """Test creating float32 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float32)
        assert vec.global_size == 100

    def test_create_i64_vector(self, context: "dtl.Context") -> None:
        """Test creating int64 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.int64)
        assert vec.global_size == 100

    def test_create_i32_vector(self, context: "dtl.Context") -> None:
        """Test creating int32 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.int32)
        assert vec.global_size == 100

    def test_create_default_dtype(self, context: "dtl.Context") -> None:
        """Test that default dtype is float64."""
        import dtl

        vec = dtl.DistributedVector(context, size=100)
        local = vec.local_view()
        assert local.dtype == np.float64

    def test_create_with_fill(self, context: "dtl.Context") -> None:
        """Test creating vector with fill value."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=42.0)
        local = vec.local_view()
        assert np.all(local == 42.0)

    def test_unsupported_dtype_raises(self, context: "dtl.Context") -> None:
        """Test that unsupported dtype raises TypeError."""
        import dtl

        with pytest.raises(TypeError):
            dtl.DistributedVector(context, size=100, dtype=np.complex128)


class TestDistributedVectorProperties:
    """Tests for DistributedVector properties."""

    def test_global_size(self, context: "dtl.Context") -> None:
        """Test global_size property."""
        import dtl

        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert vec.global_size == 1000

    def test_local_size(self, context: "dtl.Context") -> None:
        """Test local_size property."""
        import dtl

        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert vec.local_size > 0
        assert vec.local_size <= vec.global_size

    def test_local_offset(self, context: "dtl.Context") -> None:
        """Test local_offset property."""
        import dtl

        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert vec.local_offset >= 0
        assert vec.local_offset < vec.global_size

    def test_len(self, context: "dtl.Context") -> None:
        """Test __len__ returns global_size."""
        import dtl

        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert len(vec) == 1000


class TestDistributedVectorLocalView:
    """Tests for DistributedVector local_view."""

    def test_local_view_returns_numpy_array(self, context: "dtl.Context") -> None:
        """Test that local_view returns a numpy array."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        assert isinstance(local, np.ndarray)

    def test_local_view_correct_dtype(self, context: "dtl.Context") -> None:
        """Test that local_view has correct dtype."""
        import dtl

        for dtype in [np.float64, np.float32, np.int64, np.int32]:
            vec = dtl.DistributedVector(context, size=100, dtype=dtype)
            local = vec.local_view()
            assert local.dtype == dtype

    def test_local_view_correct_size(self, context: "dtl.Context") -> None:
        """Test that local_view has correct size."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        assert len(local) == vec.local_size

    def test_local_view_is_writable(self, context: "dtl.Context") -> None:
        """Test that local_view is writable (zero-copy)."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        local[:] = 123.0

        # Get view again and verify modification persisted
        local2 = vec.local_view()
        assert np.all(local2 == 123.0)

    def test_local_view_shares_memory(self, context: "dtl.Context") -> None:
        """Test that local_view shares memory with vector (zero-copy)."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local1 = vec.local_view()
        local2 = vec.local_view()

        # Modify through one view
        local1[0] = 999.0

        # Should be visible in other view
        assert local2[0] == 999.0


class TestDistributedVectorFill:
    """Tests for DistributedVector fill operation."""

    def test_fill_f64(self, context: "dtl.Context") -> None:
        """Test fill for float64 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        vec.fill(42.5)
        local = vec.local_view()
        assert np.all(local == 42.5)

    def test_fill_i32(self, context: "dtl.Context") -> None:
        """Test fill for int32 vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=100, dtype=np.int32)
        vec.fill(42)
        local = vec.local_view()
        assert np.all(local == 42)


class TestDistributedSpan:
    """Tests for DistributedSpan creation and operations."""

    def test_create_from_vector(self, context: "dtl.Context") -> None:
        """Test creating a span from distributed vector."""
        import dtl

        vec = dtl.DistributedVector(context, size=64, dtype=np.float64, fill=1.5)
        span = dtl.DistributedSpan(vec)

        assert span.global_size == vec.global_size
        assert span.local_size == vec.local_size
        assert span.rank == context.rank
        assert span.num_ranks == context.size

        local = span.local_view()
        assert local.dtype == np.float64
        assert len(local) == span.local_size
        if span.local_size > 0:
            assert local[0] == pytest.approx(1.5)

    def test_create_from_array(self, context: "dtl.Context") -> None:
        """Test creating a span from distributed array."""
        import dtl

        arr = dtl.DistributedArray(context, size=32, dtype=np.int32, fill=7)
        span = dtl.DistributedSpan(arr)

        assert span.global_size == arr.global_size
        assert span.local_size == arr.local_size
        local = span.local_view()
        assert local.dtype == np.int32
        if span.local_size > 0:
            assert int(local[0]) == 7

    def test_create_from_tensor(self, context: "dtl.Context") -> None:
        """Test creating a span from distributed tensor."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(16, 4), dtype=np.float32, fill=2.0)
        span = dtl.DistributedSpan(tensor)

        assert span.global_size == tensor.global_size
        assert span.local_size == tensor.local_size
        local = span.local_view()
        assert local.dtype == np.float32
        assert local.ndim == 1

    def test_set_local_mutates_owner_data(self, context: "dtl.Context") -> None:
        """Test span local writes are reflected in owner container."""
        import dtl

        vec = dtl.DistributedVector(context, size=32, dtype=np.float64, fill=0.0)
        span = dtl.DistributedSpan(vec)

        if span.local_size == 0:
            return

        span.set_local(0, 42.0)
        owner_local = vec.local_view()
        assert owner_local[0] == pytest.approx(42.0)

    def test_subspan_operations(self, context: "dtl.Context") -> None:
        """Test first/last/subspan operations."""
        import dtl

        vec = dtl.DistributedVector(context, size=48, dtype=np.int64, fill=5)
        span = dtl.DistributedSpan(vec)

        if span.local_size < 2:
            return

        first = span.first(1)
        assert first.local_size == 1

        last = span.last(1)
        assert last.local_size == 1

        sub_count = min(2, span.local_size - 1)
        sub = span.subspan(1, sub_count)
        assert sub.local_size == sub_count

    def test_to_from_numpy_roundtrip(self, context: "dtl.Context") -> None:
        """Test copying span data to/from NumPy arrays."""
        import dtl

        vec = dtl.DistributedVector(context, size=40, dtype=np.int32, fill=0)
        span = dtl.DistributedSpan(vec)

        local_size = span.local_size
        data = np.arange(local_size, dtype=np.int32)
        span.from_numpy(data)

        out = span.to_numpy()
        np.testing.assert_array_equal(out, data)


class TestDistributedTensorCreation:
    """Tests for DistributedTensor creation."""

    def test_create_2d_tensor(self, context: "dtl.Context") -> None:
        """Test creating 2D tensor."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(100, 64), dtype=np.float64)
        assert tensor.shape == (100, 64)
        assert tensor.ndim == 2

    def test_create_3d_tensor(self, context: "dtl.Context") -> None:
        """Test creating 3D tensor."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20, 30), dtype=np.float64)
        assert tensor.shape == (10, 20, 30)
        assert tensor.ndim == 3

    def test_create_with_fill(self, context: "dtl.Context") -> None:
        """Test creating tensor with fill value."""
        import dtl

        tensor = dtl.DistributedTensor(
            context, shape=(10, 10), dtype=np.float64, fill=3.14
        )
        local = tensor.local_view()
        assert np.all(local == 3.14)

    def test_create_i32_tensor(self, context: "dtl.Context") -> None:
        """Test creating int32 tensor."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 10), dtype=np.int32)
        local = tensor.local_view()
        assert local.dtype == np.int32


class TestDistributedTensorProperties:
    """Tests for DistributedTensor properties."""

    def test_ndim(self, context: "dtl.Context") -> None:
        """Test ndim property."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20, 30), dtype=np.float64)
        assert tensor.ndim == 3

    def test_shape(self, context: "dtl.Context") -> None:
        """Test shape property returns global shape."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(100, 64), dtype=np.float64)
        assert tensor.shape == (100, 64)

    def test_local_shape(self, context: "dtl.Context") -> None:
        """Test local_shape property."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(100, 64), dtype=np.float64)
        local_shape = tensor.local_shape

        # First dimension is distributed, others preserved
        assert len(local_shape) == 2
        assert local_shape[0] <= 100
        assert local_shape[1] == 64

    def test_global_size(self, context: "dtl.Context") -> None:
        """Test global_size property."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        assert tensor.global_size == 10 * 20

    def test_local_size(self, context: "dtl.Context") -> None:
        """Test local_size property."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        # local_size should equal product of local_shape
        local_shape = tensor.local_shape
        expected = 1
        for dim in local_shape:
            expected *= dim
        assert tensor.local_size == expected


class TestDistributedTensorLocalView:
    """Tests for DistributedTensor local_view."""

    def test_local_view_returns_numpy_array(self, context: "dtl.Context") -> None:
        """Test that local_view returns a numpy array."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        local = tensor.local_view()
        assert isinstance(local, np.ndarray)

    def test_local_view_correct_shape(self, context: "dtl.Context") -> None:
        """Test that local_view has correct shape."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        local = tensor.local_view()
        assert local.shape == tensor.local_shape

    def test_local_view_is_writable(self, context: "dtl.Context") -> None:
        """Test that local_view is writable."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        local = tensor.local_view()
        local[:] = 42.0

        local2 = tensor.local_view()
        assert np.all(local2 == 42.0)


class TestDistributedTensorFill:
    """Tests for DistributedTensor fill operation."""

    def test_fill(self, context: "dtl.Context") -> None:
        """Test fill operation."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        tensor.fill(99.0)
        local = tensor.local_view()
        assert np.all(local == 99.0)


class TestDistributedTensorToFromNumpy:
    """Tests for DistributedTensor to_numpy / from_numpy (Phase 08 parity)."""

    def test_to_numpy_returns_copy(self, context: "dtl.Context") -> None:
        """Test that to_numpy returns a copy, not a view."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64, fill=3.14)
        arr = tensor.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert np.all(arr == 3.14)

        # Modify the copy — original tensor should not change
        arr[:] = 0.0
        local = tensor.local_view()
        assert np.all(local == 3.14)

    def test_to_numpy_correct_shape(self, context: "dtl.Context") -> None:
        """Test that to_numpy has correct local shape."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(10, 20), dtype=np.float64)
        arr = tensor.to_numpy()
        assert arr.shape == tensor.local_shape

    def test_from_numpy_roundtrip(self, context: "dtl.Context") -> None:
        """Test copying data from NumPy array into tensor."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(4, 8), dtype=np.float64)
        local_shape = tensor.local_shape
        local_size = tensor.local_size

        data = np.arange(local_size, dtype=np.float64).reshape(local_shape)
        tensor.from_numpy(data)

        result = tensor.to_numpy()
        np.testing.assert_array_equal(result, data)

    def test_from_numpy_wrong_size_raises(self, context: "dtl.Context") -> None:
        """Test that from_numpy raises on size mismatch."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(4, 8), dtype=np.float64)
        wrong = np.zeros(3, dtype=np.float64)
        with pytest.raises(RuntimeError):
            tensor.from_numpy(wrong)

    def test_to_numpy_int32(self, context: "dtl.Context") -> None:
        """Test to_numpy with int32 dtype."""
        import dtl

        tensor = dtl.DistributedTensor(context, shape=(6, 10), dtype=np.int32, fill=42)
        arr = tensor.to_numpy()
        assert arr.dtype == np.int32
        assert np.all(arr == 42)


# =============================================================================
# DistributedArray Tests (Phase 10A)
# =============================================================================


class TestDistributedArrayCreation:
    """Tests for DistributedArray creation."""

    def test_create_f64_array(self, context: "dtl.Context") -> None:
        """Test creating float64 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        assert arr.global_size == 100
        assert arr.local_size > 0
        assert arr.local_size <= 100

    def test_create_f32_array(self, context: "dtl.Context") -> None:
        """Test creating float32 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float32)
        assert arr.global_size == 100

    def test_create_i64_array(self, context: "dtl.Context") -> None:
        """Test creating int64 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.int64)
        assert arr.global_size == 100

    def test_create_i32_array(self, context: "dtl.Context") -> None:
        """Test creating int32 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.int32)
        assert arr.global_size == 100

    def test_create_default_dtype(self, context: "dtl.Context") -> None:
        """Test that default dtype is float64."""
        import dtl

        arr = dtl.DistributedArray(context, size=100)
        local = arr.local_view()
        assert local.dtype == np.float64

    def test_create_with_fill(self, context: "dtl.Context") -> None:
        """Test creating array with fill value."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64, fill=42.0)
        local = arr.local_view()
        assert np.all(local == 42.0)

    def test_unsupported_dtype_raises(self, context: "dtl.Context") -> None:
        """Test that unsupported dtype raises TypeError."""
        import dtl

        with pytest.raises(TypeError):
            dtl.DistributedArray(context, size=100, dtype=np.complex128)


class TestDistributedArrayProperties:
    """Tests for DistributedArray properties."""

    def test_global_size(self, context: "dtl.Context") -> None:
        """Test global_size property."""
        import dtl

        arr = dtl.DistributedArray(context, size=1000, dtype=np.float64)
        assert arr.global_size == 1000

    def test_local_size(self, context: "dtl.Context") -> None:
        """Test local_size property."""
        import dtl

        arr = dtl.DistributedArray(context, size=1000, dtype=np.float64)
        assert arr.local_size > 0
        assert arr.local_size <= arr.global_size

    def test_local_offset(self, context: "dtl.Context") -> None:
        """Test local_offset property."""
        import dtl

        arr = dtl.DistributedArray(context, size=1000, dtype=np.float64)
        assert arr.local_offset >= 0
        assert arr.local_offset < arr.global_size

    def test_len(self, context: "dtl.Context") -> None:
        """Test __len__ returns global_size."""
        import dtl

        arr = dtl.DistributedArray(context, size=1000, dtype=np.float64)
        assert len(arr) == 1000


class TestDistributedArrayLocalView:
    """Tests for DistributedArray local_view."""

    def test_local_view_returns_numpy_array(self, context: "dtl.Context") -> None:
        """Test that local_view returns a numpy array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        local = arr.local_view()
        assert isinstance(local, np.ndarray)

    def test_local_view_correct_dtype(self, context: "dtl.Context") -> None:
        """Test that local_view has correct dtype."""
        import dtl

        for dtype in [np.float64, np.float32, np.int64, np.int32]:
            arr = dtl.DistributedArray(context, size=100, dtype=dtype)
            local = arr.local_view()
            assert local.dtype == dtype

    def test_local_view_correct_size(self, context: "dtl.Context") -> None:
        """Test that local_view has correct size."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        local = arr.local_view()
        assert len(local) == arr.local_size

    def test_local_view_is_writable(self, context: "dtl.Context") -> None:
        """Test that local_view is writable (zero-copy)."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        local = arr.local_view()
        local[:] = 123.0

        # Get view again and verify modification persisted
        local2 = arr.local_view()
        assert np.all(local2 == 123.0)

    def test_local_view_shares_memory(self, context: "dtl.Context") -> None:
        """Test that local_view shares memory with array (zero-copy)."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        local1 = arr.local_view()
        local2 = arr.local_view()

        # Modify through one view
        local1[0] = 999.0

        # Should be visible in other view
        assert local2[0] == 999.0


class TestDistributedArrayFill:
    """Tests for DistributedArray fill operation."""

    def test_fill_f64(self, context: "dtl.Context") -> None:
        """Test fill for float64 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        arr.fill(42.5)
        local = arr.local_view()
        assert np.all(local == 42.5)

    def test_fill_i32(self, context: "dtl.Context") -> None:
        """Test fill for int32 array."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.int32)
        arr.fill(42)
        local = arr.local_view()
        assert np.all(local == 42)


class TestDistributedArrayVsVector:
    """Tests documenting differences between Array and Vector."""

    def test_array_size_is_fixed(self, context: "dtl.Context") -> None:
        """Test that array size cannot be changed (no resize method)."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)

        # DistributedArray should not have resize method
        assert not hasattr(arr, 'resize')

    def test_repr_shows_fixed(self, context: "dtl.Context") -> None:
        """Test that repr indicates fixed size."""
        import dtl

        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        repr_str = repr(arr)
        assert "fixed" in repr_str.lower()
