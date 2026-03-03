# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings policy parameters in factory functions.

Phase 05 (P05): Python API Policy Parity

Tests cover:
- Factory signature accepts policy parameters
- Policies are plumbed through to C API
- Device-only placement safety (local_view raises, to_numpy works)
- Invalid/unsupported placements are rejected
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestVectorFactoryPolicies:
    """Tests for DistributedVector factory policy parameters."""

    def test_accepts_partition_parameter(self, context: "dtl.Context") -> None:
        """Test that partition parameter is accepted."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            partition=dtl.PARTITION_BLOCK
        )
        assert vec.global_size == 100

    def test_accepts_cyclic_partition(self, context: "dtl.Context") -> None:
        """Test cyclic partition parameter."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            partition=dtl.PARTITION_CYCLIC
        )
        assert vec.global_size == 100

    def test_accepts_placement_host(self, context: "dtl.Context") -> None:
        """Test host placement parameter."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        assert vec.global_size == 100
        assert vec.placement == dtl.PLACEMENT_HOST
        assert vec.is_host_accessible

    def test_accepts_execution_parameter(self, context: "dtl.Context") -> None:
        """Test execution policy parameter."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            execution=dtl.EXEC_SEQ
        )
        assert vec.global_size == 100

    def test_accepts_device_id_parameter(self, context: "dtl.Context") -> None:
        """Test device_id parameter."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            device_id=0
        )
        assert vec.global_size == 100

    def test_accepts_block_size_parameter(self, context: "dtl.Context") -> None:
        """Test block_size parameter."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            partition=dtl.PARTITION_BLOCK_CYCLIC,
            block_size=4
        )
        assert vec.global_size == 100

    def test_accepts_fill_with_policies(self, context: "dtl.Context") -> None:
        """Test fill parameter works with policy parameters."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            fill=42.0,
            partition=dtl.PARTITION_BLOCK,
            placement=dtl.PLACEMENT_HOST
        )
        local = vec.local_view()
        assert np.all(local == 42.0)

    def test_backwards_compatibility(self, context: "dtl.Context") -> None:
        """Test that old-style calls still work."""
        import dtl

        # Old-style (positional args only)
        vec = dtl.DistributedVector(context, 100, np.float64, 42.0)
        local = vec.local_view()
        assert np.all(local == 42.0)


class TestVectorHostPlacement:
    """Tests for DistributedVector with host placement."""

    def test_host_local_view_works(self, context: "dtl.Context") -> None:
        """Test local_view works for host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        local = vec.local_view()
        assert isinstance(local, np.ndarray)
        assert len(local) == vec.local_size

    def test_host_to_numpy_works(self, context: "dtl.Context") -> None:
        """Test to_numpy works for host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64, fill=5.0,
            placement=dtl.PLACEMENT_HOST
        )
        arr = vec.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.all(arr == 5.0)

    def test_host_from_numpy_works(self, context: "dtl.Context") -> None:
        """Test from_numpy works for host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        source = np.full(vec.local_size, 7.0)
        vec.from_numpy(source)
        local = vec.local_view()
        assert np.all(local == 7.0)


class TestVectorUnifiedPlacement:
    """Tests for DistributedVector with unified placement (requires CUDA)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self) -> None:
        """Skip tests if CUDA is not available."""
        import dtl
        if not dtl.has_cuda():
            pytest.skip("CUDA not available")
        if not dtl.placement_available(dtl.PLACEMENT_UNIFIED):
            pytest.skip("Unified placement not available")

    def test_unified_local_view_works(self, context: "dtl.Context") -> None:
        """Test local_view works for unified placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_UNIFIED
        )
        local = vec.local_view()
        assert isinstance(local, np.ndarray)
        local[:] = 42.0
        assert np.all(local == 42.0)

    def test_unified_is_host_accessible(self, context: "dtl.Context") -> None:
        """Test unified placement is marked as host accessible."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_UNIFIED
        )
        assert vec.is_host_accessible


class TestVectorDevicePlacement:
    """Tests for DistributedVector with device-only placement (requires CUDA)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self) -> None:
        """Skip tests if CUDA is not available."""
        import dtl
        if not dtl.has_cuda():
            pytest.skip("CUDA not available")
        if not dtl.placement_available(dtl.PLACEMENT_DEVICE):
            pytest.skip("Device placement not available")

    def test_device_local_view_raises(self, context: "dtl.Context") -> None:
        """Test local_view raises for device-only placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE
        )
        with pytest.raises(RuntimeError, match="device-only"):
            _ = vec.local_view()

    def test_device_is_not_host_accessible(self, context: "dtl.Context") -> None:
        """Test device-only is marked as not host accessible."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE
        )
        assert not vec.is_host_accessible

    def test_device_to_numpy_works(self, context: "dtl.Context") -> None:
        """Test to_numpy works for device-only placement (copies data)."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64, fill=9.0,
            placement=dtl.PLACEMENT_DEVICE
        )
        arr = vec.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.all(arr == 9.0)

    def test_device_from_numpy_works(self, context: "dtl.Context") -> None:
        """Test from_numpy works for device-only placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE
        )
        source = np.full(vec.local_size, 11.0)
        vec.from_numpy(source)

        # Verify by copying back
        arr = vec.to_numpy()
        assert np.all(arr == 11.0)


class TestPlacementAvailability:
    """Tests for placement availability checking."""

    def test_unavailable_placement_raises(self, context: "dtl.Context") -> None:
        """Test that unavailable placement raises ValueError."""
        import dtl

        # If CUDA is not available, device placements should fail
        if not dtl.has_cuda():
            with pytest.raises(ValueError, match="not available"):
                dtl.DistributedVector(
                    context, size=100,
                    placement=dtl.PLACEMENT_DEVICE
                )

    def test_host_always_available(self) -> None:
        """Test host placement is always available."""
        import dtl
        assert dtl.placement_available(dtl.PLACEMENT_HOST)


class TestArrayFactoryPolicies:
    """Tests for DistributedArray factory policy parameters."""

    def test_accepts_partition_parameter(self, context: "dtl.Context") -> None:
        """Test that partition parameter is accepted."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            partition=dtl.PARTITION_BLOCK
        )
        assert arr.global_size == 100

    def test_accepts_placement_host(self, context: "dtl.Context") -> None:
        """Test host placement parameter."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        assert arr.placement == dtl.PLACEMENT_HOST
        assert arr.is_host_accessible

    def test_local_view_works_for_host(self, context: "dtl.Context") -> None:
        """Test local_view works for host placement."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        local = arr.local_view()
        assert isinstance(local, np.ndarray)

    def test_to_numpy_works(self, context: "dtl.Context") -> None:
        """Test to_numpy for arrays."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64, fill=3.0,
            placement=dtl.PLACEMENT_HOST
        )
        result = arr.to_numpy()
        assert np.all(result == 3.0)

    def test_from_numpy_works(self, context: "dtl.Context") -> None:
        """Test from_numpy for arrays."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        source = np.full(arr.local_size, 13.0)
        arr.from_numpy(source)
        local = arr.local_view()
        assert np.all(local == 13.0)


class TestArrayDevicePlacement:
    """Tests for DistributedArray with device-only placement."""

    @pytest.fixture(autouse=True)
    def skip_if_no_cuda(self) -> None:
        """Skip tests if CUDA is not available."""
        import dtl
        if not dtl.has_cuda():
            pytest.skip("CUDA not available")
        if not dtl.placement_available(dtl.PLACEMENT_DEVICE):
            pytest.skip("Device placement not available")

    def test_device_local_view_raises(self, context: "dtl.Context") -> None:
        """Test local_view raises for device-only placement."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE
        )
        with pytest.raises(RuntimeError, match="device-only"):
            _ = arr.local_view()

    def test_device_to_numpy_works(self, context: "dtl.Context") -> None:
        """Test to_numpy works for device-only arrays."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64, fill=17.0,
            placement=dtl.PLACEMENT_DEVICE
        )
        result = arr.to_numpy()
        assert np.all(result == 17.0)


class TestFromNumpySizeValidation:
    """Tests for from_numpy size validation."""

    def test_vector_from_numpy_wrong_size_raises(self, context: "dtl.Context") -> None:
        """Test from_numpy raises on size mismatch."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        wrong_size = np.zeros(vec.local_size + 10)
        with pytest.raises(RuntimeError, match="size"):
            vec.from_numpy(wrong_size)

    def test_array_from_numpy_wrong_size_raises(self, context: "dtl.Context") -> None:
        """Test from_numpy raises on size mismatch for arrays."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        wrong_size = np.zeros(arr.local_size + 10)
        with pytest.raises(RuntimeError, match="size"):
            arr.from_numpy(wrong_size)


class TestTensorFactoryPolicies:
    """Tests for DistributedTensor factory policy parameters."""

    def test_accepts_policy_parameters(self, context: "dtl.Context") -> None:
        """Test tensor factory accepts policy parameters for API consistency."""
        import dtl

        # Tensor policies are reserved for future use, but should be accepted
        tensor = dtl.DistributedTensor(
            context, shape=(100, 10), dtype=np.float64,
            partition=dtl.PARTITION_BLOCK,
            placement=dtl.PLACEMENT_HOST
        )
        assert tensor.global_size == 1000


class TestMultiDtypePolicies:
    """Tests for policy support across multiple dtypes."""

    @pytest.mark.parametrize("dtype", [
        np.float64, np.float32, np.int64, np.int32,
        np.uint64, np.uint32, np.uint8, np.int8
    ])
    def test_vector_policies_all_dtypes(self, context: "dtl.Context", dtype) -> None:
        """Test policy parameters work for all supported dtypes."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=dtype,
            partition=dtl.PARTITION_BLOCK,
            placement=dtl.PLACEMENT_HOST,
            execution=dtl.EXEC_SEQ
        )
        assert vec.global_size == 100
        assert vec.is_host_accessible
