# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings core module.

Tests cover:
- Context creation and context manager protocol
- Rank, size, barrier operations
- mpi4py integration
- Exception handling
- Version information
"""

import pytest


class TestVersion:
    """Tests for version information."""

    def test_version_string(self) -> None:
        """Test that version string is available."""
        import dtl

        assert hasattr(dtl, "__version__")
        assert isinstance(dtl.__version__, str)
        assert len(dtl.__version__) > 0

    def test_version_info(self) -> None:
        """Test that version_info tuple is available."""
        import dtl

        assert hasattr(dtl, "version_info")
        assert isinstance(dtl.version_info, tuple)
        assert len(dtl.version_info) == 3
        # All components should be non-negative integers
        for component in dtl.version_info:
            assert isinstance(component, int)
            assert component >= 0


class TestFeatureDetection:
    """Tests for feature detection functions."""

    def test_has_mpi(self) -> None:
        """Test has_mpi function returns boolean."""
        import dtl

        result = dtl.has_mpi()
        assert isinstance(result, bool)

    def test_has_cuda(self) -> None:
        """Test has_cuda function returns boolean."""
        import dtl

        result = dtl.has_cuda()
        assert isinstance(result, bool)

    def test_has_hip(self) -> None:
        """Test has_hip function returns boolean."""
        import dtl

        result = dtl.has_hip()
        assert isinstance(result, bool)

    def test_has_nccl(self) -> None:
        """Test has_nccl function returns boolean."""
        import dtl

        result = dtl.has_nccl()
        assert isinstance(result, bool)

    def test_has_shmem(self) -> None:
        """Test has_shmem function returns boolean."""
        import dtl

        result = dtl.has_shmem()
        assert isinstance(result, bool)


class TestContextCreation:
    """Tests for Context creation."""

    def test_default_context(self) -> None:
        """Test creating context with default parameters."""
        import dtl

        ctx = dtl.Context()
        assert ctx is not None
        assert ctx.rank >= 0
        assert ctx.size >= 1
        assert ctx.device_id == -1  # Default: CPU-only

    def test_context_with_device_id(self) -> None:
        """Test creating context with explicit device_id."""
        import dtl

        # CPU-only context
        ctx = dtl.Context(device_id=-1)
        assert ctx.device_id == -1
        assert not ctx.has_device

    def test_context_manager_protocol(self) -> None:
        """Test that Context supports context manager protocol."""
        import dtl

        with dtl.Context() as ctx:
            assert ctx is not None
            rank = ctx.rank
            size = ctx.size
            assert rank >= 0
            assert size >= 1
        # Context should still be accessible after exiting
        # (cleanup is handled by destructor)


class TestContextProperties:
    """Tests for Context properties."""

    def test_rank_property(self, context: "dtl.Context") -> None:
        """Test rank property."""
        assert isinstance(context.rank, int)
        assert 0 <= context.rank < context.size

    def test_size_property(self, context: "dtl.Context") -> None:
        """Test size property."""
        assert isinstance(context.size, int)
        assert context.size >= 1

    def test_is_root_property(self, context: "dtl.Context") -> None:
        """Test is_root property."""
        assert isinstance(context.is_root, bool)
        assert context.is_root == (context.rank == 0)

    def test_device_id_property(self, context: "dtl.Context") -> None:
        """Test device_id property."""
        assert isinstance(context.device_id, int)

    def test_has_device_property(self, context: "dtl.Context") -> None:
        """Test has_device property."""
        assert isinstance(context.has_device, bool)
        # If device_id is -1, has_device should be False
        if context.device_id == -1:
            assert not context.has_device

    def test_repr(self, context: "dtl.Context") -> None:
        """Test Context string representation."""
        repr_str = repr(context)
        assert "Context" in repr_str
        assert "rank=" in repr_str
        assert "size=" in repr_str


class TestContextBarrier:
    """Tests for Context barrier operation."""

    def test_barrier(self, context: "dtl.Context") -> None:
        """Test that barrier completes without error."""
        # This should complete without raising an exception
        context.barrier()

    def test_multiple_barriers(self, context: "dtl.Context") -> None:
        """Test multiple consecutive barriers."""
        for _ in range(3):
            context.barrier()


@pytest.mark.mpi
class TestMPI4PyIntegration:
    """Tests for mpi4py integration."""

    def test_context_with_mpi_comm(self) -> None:
        """Test creating context with mpi4py communicator."""
        import dtl
        if not dtl.has_mpi():
            pytest.skip("DTL was built without MPI support")

        try:
            from mpi4py import MPI

            ctx = dtl.Context(comm=MPI.COMM_WORLD)
            assert ctx.rank == MPI.COMM_WORLD.Get_rank()
            assert ctx.size == MPI.COMM_WORLD.Get_size()
        except ImportError:
            pytest.skip("mpi4py not available")

    def test_context_with_none_comm(self) -> None:
        """Test creating context with None communicator (should use COMM_WORLD)."""
        import dtl

        ctx = dtl.Context(comm=None)
        assert ctx is not None
        assert ctx.rank >= 0
        assert ctx.size >= 1


class TestExceptions:
    """Tests for DTL exception handling."""

    def test_dtl_error_hierarchy(self) -> None:
        """Test that DTL exceptions exist and have proper hierarchy."""
        import dtl

        # All DTL exceptions should be defined
        assert hasattr(dtl, "DTLError")
        assert hasattr(dtl, "CommunicationError")
        assert hasattr(dtl, "MemoryError")
        assert hasattr(dtl, "BoundsError")
        assert hasattr(dtl, "InvalidArgumentError")
        assert hasattr(dtl, "BackendError")

        # All should inherit from DTLError
        assert issubclass(dtl.CommunicationError, dtl.DTLError)
        assert issubclass(dtl.MemoryError, dtl.DTLError)
        assert issubclass(dtl.BoundsError, dtl.DTLError)
        assert issubclass(dtl.InvalidArgumentError, dtl.DTLError)
        assert issubclass(dtl.BackendError, dtl.DTLError)

    def test_bounds_error_is_index_error(self) -> None:
        """Test that BoundsError inherits from IndexError."""
        import dtl

        assert issubclass(dtl.BoundsError, IndexError)

    def test_invalid_argument_error_is_value_error(self) -> None:
        """Test that InvalidArgumentError inherits from ValueError."""
        import dtl

        assert issubclass(dtl.InvalidArgumentError, ValueError)

    def test_memory_error_is_memory_error(self) -> None:
        """Test that DTL MemoryError inherits from builtin MemoryError."""
        import dtl

        assert issubclass(dtl.MemoryError, MemoryError)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        import dtl

        expected = [
            "__version__",
            "version_info",
            "has_mpi",
            "has_cuda",
            "has_hip",
            "has_nccl",
            "has_shmem",
            "Context",
            "DistributedVector",
            "DistributedTensor",
            "DTLError",
            "CommunicationError",
            "MemoryError",
            "BoundsError",
            "InvalidArgumentError",
            "BackendError",
        ]

        for name in expected:
            assert name in dtl.__all__, f"{name} missing from __all__"

    def test_all_exports_are_accessible(self) -> None:
        """Test that all exports in __all__ are accessible."""
        import dtl

        for name in dtl.__all__:
            assert hasattr(dtl, name), f"{name} in __all__ but not accessible"


class TestEnvironment:
    """Tests for Environment lifecycle and context factories."""

    def test_environment_creation(self) -> None:
        """Test creating an environment."""
        import dtl

        env = dtl.Environment()
        assert env is not None

    def test_environment_context_manager(self) -> None:
        """Test Environment as context manager."""
        import dtl

        with dtl.Environment() as env:
            assert env is not None

    def test_environment_is_initialized(self) -> None:
        """Test is_initialized static query."""
        import dtl

        env = dtl.Environment()
        assert dtl.Environment.is_initialized()
        del env

    def test_environment_ref_count(self) -> None:
        """Test ref_count tracking."""
        import dtl

        env1 = dtl.Environment()
        count1 = dtl.Environment.ref_count()
        assert count1 >= 1

        env2 = dtl.Environment()
        count2 = dtl.Environment.ref_count()
        assert count2 >= count1

        del env2
        del env1

    def test_environment_backend_queries(self) -> None:
        """Test backend availability queries."""
        import dtl

        with dtl.Environment() as env:
            # These should all return booleans
            assert isinstance(env.has_mpi, bool)
            assert isinstance(env.has_cuda, bool)
            assert isinstance(env.has_hip, bool)
            assert isinstance(env.has_nccl, bool)
            assert isinstance(env.has_shmem, bool)

    def test_environment_mpi_thread_level(self) -> None:
        """Test MPI thread level query."""
        import dtl

        with dtl.Environment() as env:
            level = dtl.Environment.mpi_thread_level()
            assert isinstance(level, int)
            assert -1 <= level <= 3

    def test_environment_make_world_context(self) -> None:
        """Test creating a world context from environment."""
        import dtl

        with dtl.Environment() as env:
            ctx = env.make_world_context()
            assert ctx is not None
            assert isinstance(ctx, dtl.Context)
            assert ctx.rank >= 0
            assert ctx.size >= 1

    def test_environment_make_cpu_context(self) -> None:
        """Test creating a CPU-only context from environment."""
        import dtl

        with dtl.Environment() as env:
            ctx = env.make_cpu_context()
            assert ctx is not None
            assert isinstance(ctx, dtl.Context)
            assert ctx.rank == 0
            assert ctx.size == 1

    def test_environment_repr(self) -> None:
        """Test Environment string representation."""
        import dtl

        with dtl.Environment() as env:
            repr_str = repr(env)
            assert "Environment" in repr_str
            assert "backends=" in repr_str
            assert "refcount=" in repr_str

    def test_environment_in_all(self) -> None:
        """Test that Environment is exported in __all__."""
        import dtl

        assert "Environment" in dtl.__all__
