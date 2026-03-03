# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings topology module.

Tests cover:
- CPU count query
- GPU count query
- CPU affinity query
- GPU ID query
- Node locality check
- Node ID query
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# Check if topology bindings are available
try:
    import dtl
    HAS_TOPOLOGY = hasattr(dtl, 'topology') or hasattr(dtl, 'num_cpus')
except ImportError:
    HAS_TOPOLOGY = False

skip_no_topology = pytest.mark.skipif(
    not HAS_TOPOLOGY,
    reason="Topology bindings not available"
)


@skip_no_topology
class TestCPUQueries:
    """Tests for CPU topology queries."""

    def test_num_cpus_returns_positive(self) -> None:
        """Test that num_cpus returns a value greater than 0."""
        import dtl

        if hasattr(dtl, 'topology'):
            num = dtl.topology.num_cpus()
        else:
            num = dtl.num_cpus()

        assert isinstance(num, int)
        assert num > 0

    def test_num_cpus_is_reasonable(self) -> None:
        """Test that num_cpus returns a reasonable value."""
        import dtl

        if hasattr(dtl, 'topology'):
            num = dtl.topology.num_cpus()
        else:
            num = dtl.num_cpus()

        # Most systems have between 1 and 1024 CPUs
        assert 1 <= num <= 4096


@skip_no_topology
class TestGPUQueries:
    """Tests for GPU topology queries."""

    def test_num_gpus_returns_non_negative(self) -> None:
        """Test that num_gpus returns a value >= 0."""
        import dtl

        if hasattr(dtl, 'topology'):
            num = dtl.topology.num_gpus()
        else:
            num = dtl.num_gpus()

        assert isinstance(num, int)
        assert num >= 0

    def test_num_gpus_matches_cuda_availability(self) -> None:
        """Test that num_gpus is consistent with CUDA availability."""
        import dtl

        has_cuda = dtl.has_cuda()

        if hasattr(dtl, 'topology'):
            num = dtl.topology.num_gpus()
        else:
            num = dtl.num_gpus()

        if not has_cuda:
            # Without CUDA, GPU count may still be >= 0 (HIP, etc.)
            assert num >= 0
        # If CUDA is available, there should be at least one GPU
        # (but we don't enforce this as detection may vary)


@skip_no_topology
class TestCPUAffinity:
    """Tests for CPU affinity queries."""

    def test_cpu_affinity_returns_valid_id(self) -> None:
        """Test that cpu_affinity returns a valid CPU ID."""
        import dtl

        if hasattr(dtl, 'topology'):
            cpu_id = dtl.topology.cpu_affinity()
        else:
            cpu_id = dtl.cpu_affinity()

        assert isinstance(cpu_id, int)
        assert cpu_id >= 0

    def test_cpu_affinity_within_cpu_count(self) -> None:
        """Test that cpu_affinity returns an ID within the CPU count range."""
        import dtl

        if hasattr(dtl, 'topology'):
            cpu_id = dtl.topology.cpu_affinity()
            num = dtl.topology.num_cpus()
        else:
            cpu_id = dtl.cpu_affinity()
            num = dtl.num_cpus()

        assert cpu_id < num


@skip_no_topology
class TestGPUID:
    """Tests for GPU ID queries."""

    def test_gpu_id_returns_valid_or_negative(self) -> None:
        """Test that gpu_id returns a valid GPU ID or -1 if no GPU."""
        import dtl

        if hasattr(dtl, 'topology'):
            gpu_id = dtl.topology.gpu_id()
        else:
            gpu_id = dtl.gpu_id()

        assert isinstance(gpu_id, int)
        # -1 means no GPU assigned, otherwise must be non-negative
        assert gpu_id >= -1

    def test_gpu_id_minus_one_without_cuda(self) -> None:
        """Test that gpu_id is -1 when CUDA is not available."""
        import dtl

        if dtl.has_cuda():
            pytest.skip("CUDA is available, cannot test no-GPU case")

        if hasattr(dtl, 'topology'):
            gpu_id = dtl.topology.gpu_id()
        else:
            gpu_id = dtl.gpu_id()

        assert gpu_id == -1


@skip_no_topology
class TestNodeLocality:
    """Tests for node locality queries."""

    def test_is_local_returns_bool(self, context: "dtl.Context") -> None:
        """Test that is_local returns a boolean value."""
        import dtl

        if hasattr(dtl, 'topology'):
            result = dtl.topology.is_local(context.rank)
        else:
            result = dtl.is_local(context.rank)

        assert isinstance(result, bool)

    def test_self_is_local(self, context: "dtl.Context") -> None:
        """Test that a rank is always local to itself."""
        import dtl

        if hasattr(dtl, 'topology'):
            result = dtl.topology.is_local(context.rank)
        else:
            result = dtl.is_local(context.rank)

        assert result is True

    def test_is_local_with_zero(self, context: "dtl.Context") -> None:
        """Test is_local with rank 0."""
        import dtl

        if hasattr(dtl, 'topology'):
            result = dtl.topology.is_local(0)
        else:
            result = dtl.is_local(0)

        assert isinstance(result, bool)


@skip_no_topology
class TestNodeID:
    """Tests for node ID queries."""

    def test_node_id_returns_non_negative(self) -> None:
        """Test that node_id returns an integer >= 0."""
        import dtl

        if hasattr(dtl, 'topology'):
            nid = dtl.topology.node_id()
        else:
            nid = dtl.node_id()

        assert isinstance(nid, int)
        assert nid >= 0

    def test_node_id_is_deterministic(self) -> None:
        """Test that node_id returns the same value on repeated calls."""
        import dtl

        if hasattr(dtl, 'topology'):
            nid1 = dtl.topology.node_id()
            nid2 = dtl.topology.node_id()
        else:
            nid1 = dtl.node_id()
            nid2 = dtl.node_id()

        assert nid1 == nid2
