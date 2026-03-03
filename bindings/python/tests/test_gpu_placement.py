# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings GPU placement support (Phase 28).

Tests cover:
- PlacementPolicy enum values are accessible
- DEVICE and UNIFIED placements require CUDA
- to_numpy() works for all placements
- from_numpy() works for all placements
- local_view() raises for device-only placement
- Host placement containers work normally
- to_device() and to_host() migration functions

All GPU-specific tests are skipped when CUDA is not available.
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# Helper to check CUDA availability at module level
def _has_cuda():
    try:
        import dtl
        return dtl.has_cuda()
    except ImportError:
        return False


requires_cuda = pytest.mark.skipif(
    not _has_cuda(),
    reason="CUDA not available"
)


class TestPlacementPolicyAccess:
    """Tests for placement policy enum accessibility."""

    def test_placement_host_exists(self) -> None:
        """Test PLACEMENT_HOST is accessible."""
        import dtl
        assert hasattr(dtl, "PLACEMENT_HOST")

    def test_placement_device_exists(self) -> None:
        """Test PLACEMENT_DEVICE is accessible."""
        import dtl
        assert hasattr(dtl, "PLACEMENT_DEVICE")

    def test_placement_unified_exists(self) -> None:
        """Test PLACEMENT_UNIFIED is accessible."""
        import dtl
        assert hasattr(dtl, "PLACEMENT_UNIFIED")

    def test_placement_device_preferred_exists(self) -> None:
        """Test PLACEMENT_DEVICE_PREFERRED is accessible."""
        import dtl
        assert hasattr(dtl, "PLACEMENT_DEVICE_PREFERRED")


class TestHostPlacement:
    """Tests for host placement (always available)."""

    def test_vector_host_placement(self, context: "dtl.Context") -> None:
        """Test vector creation with explicit host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        assert vec.global_size == 100
        assert vec.is_host_accessible is True

    def test_vector_host_local_view(self, context: "dtl.Context") -> None:
        """Test local_view works with host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64, fill=42.0,
            placement=dtl.PLACEMENT_HOST
        )
        local = vec.local_view()
        assert isinstance(local, np.ndarray)
        assert np.all(local == 42.0)

    def test_vector_host_to_numpy(self, context: "dtl.Context") -> None:
        """Test to_numpy works with host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=50, dtype=np.float64, fill=7.5,
            placement=dtl.PLACEMENT_HOST
        )
        arr = vec.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert np.all(arr == 7.5)

    def test_vector_host_from_numpy(self, context: "dtl.Context") -> None:
        """Test from_numpy works with host placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=10, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        data = np.arange(vec.local_size, dtype=np.float64)
        vec.from_numpy(data)
        local = vec.local_view()
        assert np.allclose(local, data)


@requires_cuda
class TestDevicePlacement:
    """Tests for device-only placement (requires CUDA)."""

    def test_vector_device_creation(self, context: "dtl.Context") -> None:
        """Test creating a device-only vector."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE, device_id=0
        )
        assert vec.global_size == 100
        assert vec.is_host_accessible is False

    def test_vector_device_local_view_raises(self, context: "dtl.Context") -> None:
        """Test that local_view raises for device-only placement."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_DEVICE, device_id=0
        )
        with pytest.raises(RuntimeError, match="device-only"):
            vec.local_view()


@requires_cuda
class TestUnifiedPlacement:
    """Tests for unified memory placement (requires CUDA)."""

    def test_vector_unified_creation(self, context: "dtl.Context") -> None:
        """Test creating a unified memory vector."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64,
            placement=dtl.PLACEMENT_UNIFIED
        )
        assert vec.global_size == 100
        assert vec.is_host_accessible is True


class TestDeviceMigrationAPI:
    """Tests for to_device() and to_host() functions (always importable)."""

    def test_to_device_function_exists(self) -> None:
        """Test to_device is importable from dtl."""
        import dtl
        assert hasattr(dtl, "to_device")
        assert callable(dtl.to_device)

    def test_to_host_function_exists(self) -> None:
        """Test to_host is importable from dtl."""
        import dtl
        assert hasattr(dtl, "to_host")
        assert callable(dtl.to_host)

    def test_to_host_from_host_vector(self, context: "dtl.Context") -> None:
        """Test to_host creates a host copy of a host vector."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=50, dtype=np.float64, fill=3.14,
            placement=dtl.PLACEMENT_HOST
        )
        host_copy = dtl.to_host(context, vec)
        assert host_copy.global_size == 50
        assert host_copy.is_host_accessible is True
        arr = host_copy.to_numpy()
        assert np.allclose(arr, 3.14)

    def test_to_host_from_host_array(self, context: "dtl.Context") -> None:
        """Test to_host works with DistributedArray."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=30, dtype=np.int32, fill=7,
            placement=dtl.PLACEMENT_HOST
        )
        host_copy = dtl.to_host(context, arr)
        assert host_copy.global_size == 30
        assert host_copy.is_host_accessible is True

    def test_to_host_preserves_data(self, context: "dtl.Context") -> None:
        """Test to_host preserves element values."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=20, dtype=np.float32,
            placement=dtl.PLACEMENT_HOST
        )
        data = np.arange(vec.local_size, dtype=np.float32) * 2.5
        vec.from_numpy(data)

        host_copy = dtl.to_host(context, vec)
        result = host_copy.to_numpy()
        assert np.allclose(result, data)

    def test_to_device_requires_cuda(self, context: "dtl.Context") -> None:
        """Test to_device raises if CUDA is not available."""
        import dtl

        if dtl.placement_available(dtl.PLACEMENT_DEVICE):
            pytest.skip("CUDA is available -- cannot test unavailable path")

        vec = dtl.DistributedVector(
            context, size=10, dtype=np.float64, fill=1.0,
            placement=dtl.PLACEMENT_HOST
        )
        with pytest.raises(ValueError, match="PLACEMENT_DEVICE"):
            dtl.to_device(context, vec)


@requires_cuda
class TestDeviceMigrationGPU:
    """Tests for to_device()/to_host() round-trip with actual GPU."""

    def test_to_device_creates_device_container(self, context: "dtl.Context") -> None:
        """Test to_device creates a device-placed container."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=100, dtype=np.float64, fill=42.0,
            placement=dtl.PLACEMENT_HOST
        )
        device_vec = dtl.to_device(context, vec, device_id=0)
        assert device_vec.global_size == 100
        assert device_vec.is_host_accessible is False

    def test_round_trip_host_device_host(self, context: "dtl.Context") -> None:
        """Test data survives host -> device -> host round trip."""
        import dtl

        vec = dtl.DistributedVector(
            context, size=64, dtype=np.float64,
            placement=dtl.PLACEMENT_HOST
        )
        data = np.arange(vec.local_size, dtype=np.float64)
        vec.from_numpy(data)

        # Host -> Device
        device_vec = dtl.to_device(context, vec)

        # Device -> Host
        host_vec = dtl.to_host(context, device_vec)
        result = host_vec.to_numpy()
        assert np.allclose(result, data)

    def test_to_device_array_round_trip(self, context: "dtl.Context") -> None:
        """Test round trip with DistributedArray."""
        import dtl

        arr = dtl.DistributedArray(
            context, size=32, dtype=np.float32, fill=2.718,
            placement=dtl.PLACEMENT_HOST
        )
        device_arr = dtl.to_device(context, arr)
        host_arr = dtl.to_host(context, device_arr)
        result = host_arr.to_numpy()
        assert np.allclose(result, 2.718, rtol=1e-5)
