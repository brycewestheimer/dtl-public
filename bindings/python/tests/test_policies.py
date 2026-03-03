# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings policy module.

Tests cover:
- Partition policy enum and constants
- Placement policy enum and constants
- Execution policy enum and constants
- Policy availability checking
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestPartitionPolicy:
    """Tests for partition policy enum and constants."""

    def test_partition_block_defined(self) -> None:
        """Test PARTITION_BLOCK is defined."""
        import dtl

        assert hasattr(dtl, 'PARTITION_BLOCK')
        assert dtl.PARTITION_BLOCK == 0

    def test_partition_cyclic_defined(self) -> None:
        """Test PARTITION_CYCLIC is defined."""
        import dtl

        assert hasattr(dtl, 'PARTITION_CYCLIC')
        assert dtl.PARTITION_CYCLIC == 1

    def test_partition_block_cyclic_defined(self) -> None:
        """Test PARTITION_BLOCK_CYCLIC is defined."""
        import dtl

        assert hasattr(dtl, 'PARTITION_BLOCK_CYCLIC')
        assert dtl.PARTITION_BLOCK_CYCLIC == 2

    def test_partition_hash_defined(self) -> None:
        """Test PARTITION_HASH is defined."""
        import dtl

        assert hasattr(dtl, 'PARTITION_HASH')
        assert dtl.PARTITION_HASH == 3

    def test_partition_replicated_defined(self) -> None:
        """Test PARTITION_REPLICATED is defined."""
        import dtl

        assert hasattr(dtl, 'PARTITION_REPLICATED')
        assert dtl.PARTITION_REPLICATED == 4

    def test_partition_policy_enum_exists(self) -> None:
        """Test PartitionPolicy enum is exported."""
        import dtl

        assert hasattr(dtl, 'PartitionPolicy')


class TestPlacementPolicy:
    """Tests for placement policy enum and constants."""

    def test_placement_host_defined(self) -> None:
        """Test PLACEMENT_HOST is defined."""
        import dtl

        assert hasattr(dtl, 'PLACEMENT_HOST')
        assert dtl.PLACEMENT_HOST == 0

    def test_placement_device_defined(self) -> None:
        """Test PLACEMENT_DEVICE is defined."""
        import dtl

        assert hasattr(dtl, 'PLACEMENT_DEVICE')
        assert dtl.PLACEMENT_DEVICE == 1

    def test_placement_unified_defined(self) -> None:
        """Test PLACEMENT_UNIFIED is defined."""
        import dtl

        assert hasattr(dtl, 'PLACEMENT_UNIFIED')
        assert dtl.PLACEMENT_UNIFIED == 2

    def test_placement_device_preferred_defined(self) -> None:
        """Test PLACEMENT_DEVICE_PREFERRED is defined."""
        import dtl

        assert hasattr(dtl, 'PLACEMENT_DEVICE_PREFERRED')
        assert dtl.PLACEMENT_DEVICE_PREFERRED == 3

    def test_placement_policy_enum_exists(self) -> None:
        """Test PlacementPolicy enum is exported."""
        import dtl

        assert hasattr(dtl, 'PlacementPolicy')


class TestExecutionPolicy:
    """Tests for execution policy enum and constants."""

    def test_exec_seq_defined(self) -> None:
        """Test EXEC_SEQ is defined."""
        import dtl

        assert hasattr(dtl, 'EXEC_SEQ')
        assert dtl.EXEC_SEQ == 0

    def test_exec_par_defined(self) -> None:
        """Test EXEC_PAR is defined."""
        import dtl

        assert hasattr(dtl, 'EXEC_PAR')
        assert dtl.EXEC_PAR == 1

    def test_exec_async_defined(self) -> None:
        """Test EXEC_ASYNC is defined."""
        import dtl

        assert hasattr(dtl, 'EXEC_ASYNC')
        assert dtl.EXEC_ASYNC == 2

    def test_execution_policy_enum_exists(self) -> None:
        """Test ExecutionPolicy enum is exported."""
        import dtl

        assert hasattr(dtl, 'ExecutionPolicy')


class TestPlacementAvailability:
    """Tests for placement availability checking."""

    def test_host_always_available(self) -> None:
        """Test that host placement is always available."""
        import dtl

        assert dtl.placement_available(dtl.PLACEMENT_HOST) is True

    def test_device_available_matches_cuda(self) -> None:
        """Test that device placement matches CUDA availability."""
        import dtl

        device_available = dtl.placement_available(dtl.PLACEMENT_DEVICE)
        cuda_available = dtl.has_cuda()

        # Device should only be available if CUDA is available
        if cuda_available:
            assert device_available is True
        else:
            assert device_available is False

    def test_unified_available_matches_cuda(self) -> None:
        """Test that unified placement matches CUDA availability."""
        import dtl

        unified_available = dtl.placement_available(dtl.PLACEMENT_UNIFIED)
        cuda_available = dtl.has_cuda()

        if cuda_available:
            assert unified_available is True
        else:
            assert unified_available is False

    def test_device_preferred_available_matches_cuda(self) -> None:
        """Test that device_preferred placement matches CUDA availability."""
        import dtl

        dp_available = dtl.placement_available(dtl.PLACEMENT_DEVICE_PREFERRED)
        cuda_available = dtl.has_cuda()

        if cuda_available:
            assert dp_available is True
        else:
            assert dp_available is False


class TestPolicyWithContainers:
    """Tests for using policies with containers."""

    def test_vector_with_default_policies(self, context: "dtl.Context") -> None:
        """Test creating vector with default policies."""
        import dtl
        import numpy as np

        # Standard creation uses default policies
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        assert vec.global_size == 100

    def test_array_with_default_policies(self, context: "dtl.Context") -> None:
        """Test creating array with default policies."""
        import dtl
        import numpy as np

        # Standard creation uses default policies
        arr = dtl.DistributedArray(context, size=100, dtype=np.float64)
        assert arr.global_size == 100

    # Note: Policy-aware creation with custom policies would require
    # additional parameters in the factory functions, which is planned
    # for a future update.
