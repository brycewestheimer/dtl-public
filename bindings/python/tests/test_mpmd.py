# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings MPMD (Multiple Program Multiple Data) module.

Tests cover:
- RoleManager creation and destruction
- Role registration and initialization
- Role query operations (has_role, role_size, role_rank)
- Inter-group communication (intergroup_send, intergroup_recv)
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# Check if MPMD bindings are available
try:
    import dtl
    HAS_MPMD = hasattr(dtl, 'RoleManager')
except ImportError:
    HAS_MPMD = False

skip_no_mpmd = pytest.mark.skipif(
    not HAS_MPMD,
    reason="MPMD bindings not available"
)


@skip_no_mpmd
class TestRoleManagerCreation:
    """Tests for RoleManager creation and destruction."""

    def test_create_default(self) -> None:
        """Test creating RoleManager with default configuration."""
        import dtl

        manager = dtl.RoleManager()
        assert manager is not None

    def test_create_and_destroy(self) -> None:
        """Test that RoleManager can be created and garbage collected."""
        import dtl

        manager = dtl.RoleManager()
        assert manager is not None
        del manager  # Should not raise

    def test_not_initialized_by_default(self) -> None:
        """Test that RoleManager is not initialized after construction."""
        import dtl

        manager = dtl.RoleManager()
        assert not manager.initialized()

    def test_repr(self) -> None:
        """Test RoleManager string representation."""
        import dtl

        manager = dtl.RoleManager()
        repr_str = repr(manager)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


@skip_no_mpmd
class TestRoleRegistration:
    """Tests for role registration via add_role."""

    def test_add_role(self) -> None:
        """Test adding a role to the manager."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: rank > 0)

    def test_add_multiple_roles(self) -> None:
        """Test adding multiple roles."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("coordinator", lambda rank, size: rank == 0)
        manager.add_role("worker", lambda rank, size: rank > 0)

    def test_add_role_with_string_name(self) -> None:
        """Test that role name must be a string."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("io_handler", lambda rank, size: rank == size - 1)

    def test_add_role_after_initialize_raises(self, context: "dtl.Context") -> None:
        """Test that adding a role after initialization raises an error."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        with pytest.raises((RuntimeError, dtl.DTLError)):
            manager.add_role("late_role", lambda rank, size: True)


@skip_no_mpmd
class TestRoleManagerInitialization:
    """Tests for RoleManager initialization."""

    def test_initialize(self, context: "dtl.Context") -> None:
        """Test initializing the role manager."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)
        assert manager.initialized()

    def test_double_initialize_raises(self, context: "dtl.Context") -> None:
        """Test that initializing twice raises an error."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        with pytest.raises((RuntimeError, dtl.DTLError)):
            manager.initialize(context)

    def test_initialize_with_no_roles(self, context: "dtl.Context") -> None:
        """Test initializing with no registered roles."""
        import dtl

        manager = dtl.RoleManager()
        # Should either succeed (with default role) or raise
        try:
            manager.initialize(context)
        except (RuntimeError, dtl.DTLError):
            pass  # Acceptable behavior


@skip_no_mpmd
class TestRoleQueries:
    """Tests for role query operations."""

    def test_has_role(self, context: "dtl.Context") -> None:
        """Test has_role returns correct result."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        assert manager.has_role("worker") is True

    def test_has_role_not_assigned(self, context: "dtl.Context") -> None:
        """Test has_role returns False for unassigned role."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("coordinator", lambda rank, size: False)
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        assert manager.has_role("coordinator") is False
        assert manager.has_role("worker") is True

    def test_role_size(self, context: "dtl.Context") -> None:
        """Test role_size returns the number of ranks with a given role."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        role_size = manager.role_size("worker")
        assert isinstance(role_size, int)
        assert role_size >= 1

    def test_role_rank(self, context: "dtl.Context") -> None:
        """Test role_rank returns this rank's index within its role group."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        role_rank = manager.role_rank("worker")
        assert isinstance(role_rank, int)
        assert role_rank >= 0

    def test_role_rank_less_than_role_size(self, context: "dtl.Context") -> None:
        """Test that role_rank < role_size for assigned roles."""
        import dtl

        manager = dtl.RoleManager()
        manager.add_role("worker", lambda rank, size: True)
        manager.initialize(context)

        assert manager.role_rank("worker") < manager.role_size("worker")


@skip_no_mpmd
@pytest.mark.mpi
class TestIntergroupCommunication:
    """Tests for inter-group communication operations."""

    def test_intergroup_send_recv(self, mpi_context: "dtl.Context") -> None:
        """Test intergroup send and receive between roles."""
        import dtl
        import numpy as np

        ctx = mpi_context
        if ctx.size < 2:
            pytest.skip("Requires at least 2 MPI ranks")

        manager = dtl.RoleManager()
        manager.add_role("sender", lambda rank, size: rank == 0)
        manager.add_role("receiver", lambda rank, size: rank == 1)
        manager.initialize(ctx)

        data = np.array([42.0], dtype=np.float64)

        if manager.has_role("sender"):
            dtl.intergroup_send(manager, "receiver", data, tag=0)
        elif manager.has_role("receiver"):
            received = dtl.intergroup_recv(manager, "sender", dtype=np.float64, count=1, tag=0)
            np.testing.assert_array_equal(received, data)

    def test_intergroup_send_recv_array(self, mpi_context: "dtl.Context") -> None:
        """Test intergroup send/receive with larger arrays."""
        import dtl
        import numpy as np

        ctx = mpi_context
        if ctx.size < 2:
            pytest.skip("Requires at least 2 MPI ranks")

        manager = dtl.RoleManager()
        manager.add_role("producer", lambda rank, size: rank == 0)
        manager.add_role("consumer", lambda rank, size: rank == 1)
        manager.initialize(ctx)

        data = np.arange(10, dtype=np.int64)

        if manager.has_role("producer"):
            dtl.intergroup_send(manager, "consumer", data, tag=1)
        elif manager.has_role("consumer"):
            received = dtl.intergroup_recv(manager, "producer", dtype=np.int64, count=10, tag=1)
            np.testing.assert_array_equal(received, data)
