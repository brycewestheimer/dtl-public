# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings remote actions module.

Tests cover:
- Action registration
- remote_invoke on same rank (synchronous)
- remote_invoke_async on same rank (asynchronous)
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# Check if remote bindings are available
try:
    import dtl
    HAS_REMOTE = hasattr(dtl, 'Action') or hasattr(dtl, 'remote_invoke')
except ImportError:
    HAS_REMOTE = False

skip_no_remote = pytest.mark.skipif(
    not HAS_REMOTE,
    reason="Remote action bindings not available"
)


@skip_no_remote
class TestActionRegistration:
    """Tests for Action registration."""

    def test_register_action(self) -> None:
        """Test registering a simple action."""
        import dtl

        def add(a, b):
            return a + b

        action = dtl.Action("add", add)
        assert action is not None

    def test_register_action_with_name(self) -> None:
        """Test that registered action has a name."""
        import dtl

        def multiply(a, b):
            return a * b

        action = dtl.Action("multiply", multiply)
        assert action.name == "multiply"

    def test_register_void_action(self) -> None:
        """Test registering an action with no return value."""
        import dtl

        results = []

        def append_value(x):
            results.append(x)

        action = dtl.Action("append_value", append_value)
        assert action is not None

    def test_register_multiple_actions(self) -> None:
        """Test registering multiple distinct actions."""
        import dtl

        action1 = dtl.Action("op1", lambda x: x + 1)
        action2 = dtl.Action("op2", lambda x: x * 2)
        action3 = dtl.Action("op3", lambda x: x ** 2)

        assert action1.name == "op1"
        assert action2.name == "op2"
        assert action3.name == "op3"

    def test_register_action_callable(self) -> None:
        """Test that action wraps a callable."""
        import dtl

        action = dtl.Action("identity", lambda x: x)
        assert callable(action) or hasattr(action, 'invoke')


@skip_no_remote
class TestRemoteInvoke:
    """Tests for synchronous remote_invoke on same rank."""

    def test_invoke_on_self(self, context: "dtl.Context") -> None:
        """Test invoking a remote action on the calling rank."""
        import dtl

        def add(a, b):
            return a + b

        action = dtl.Action("add", add)

        result = dtl.remote_invoke(context, action, context.rank, 3, 4)
        assert result == 7

    def test_invoke_with_single_arg(self, context: "dtl.Context") -> None:
        """Test invoking with a single argument."""
        import dtl

        def double(x):
            return x * 2

        action = dtl.Action("double", double)

        result = dtl.remote_invoke(context, action, context.rank, 21)
        assert result == 42

    def test_invoke_with_no_args(self, context: "dtl.Context") -> None:
        """Test invoking an action with no arguments."""
        import dtl

        def get_answer():
            return 42

        action = dtl.Action("get_answer", get_answer)

        result = dtl.remote_invoke(context, action, context.rank)
        assert result == 42

    def test_invoke_void_action(self, context: "dtl.Context") -> None:
        """Test invoking an action that returns None."""
        import dtl

        side_effects = []

        def record(val):
            side_effects.append(val)

        action = dtl.Action("record", record)

        result = dtl.remote_invoke(context, action, context.rank, 99)
        assert result is None

    def test_invoke_with_float_args(self, context: "dtl.Context") -> None:
        """Test invoking with float arguments."""
        import dtl

        def multiply(a, b):
            return a * b

        action = dtl.Action("multiply", multiply)

        result = dtl.remote_invoke(context, action, context.rank, 2.5, 4.0)
        assert abs(result - 10.0) < 1e-10

    def test_invoke_with_string_arg(self, context: "dtl.Context") -> None:
        """Test invoking with a string argument."""
        import dtl

        def greet(name):
            return f"Hello, {name}!"

        action = dtl.Action("greet", greet)

        result = dtl.remote_invoke(context, action, context.rank, "DTL")
        assert result == "Hello, DTL!"


@skip_no_remote
class TestRemoteInvokeAsync:
    """Tests for asynchronous remote_invoke_async on same rank."""

    def test_invoke_async_on_self(self, context: "dtl.Context") -> None:
        """Test async invocation on the calling rank."""
        import dtl

        def add(a, b):
            return a + b

        action = dtl.Action("add_async", add)

        future = dtl.remote_invoke_async(context, action, context.rank, 10, 20)
        assert future is not None

        result = future.get()
        assert result == 30

    def test_invoke_async_returns_future(self, context: "dtl.Context") -> None:
        """Test that async invocation returns a future-like object."""
        import dtl

        def identity(x):
            return x

        action = dtl.Action("identity_async", identity)

        future = dtl.remote_invoke_async(context, action, context.rank, 42)

        # Future should have get() method
        assert hasattr(future, 'get')
        assert future.get() == 42

    def test_invoke_async_with_wait(self, context: "dtl.Context") -> None:
        """Test waiting on an async invocation result."""
        import dtl

        def compute(x):
            return x * x

        action = dtl.Action("compute_async", compute)

        future = dtl.remote_invoke_async(context, action, context.rank, 7)

        # Wait should complete without error
        if hasattr(future, 'wait'):
            future.wait()

        assert future.get() == 49

    def test_invoke_async_test_readiness(self, context: "dtl.Context") -> None:
        """Test checking readiness of an async invocation."""
        import dtl

        def noop():
            return None

        action = dtl.Action("noop_async", noop)

        future = dtl.remote_invoke_async(context, action, context.rank)

        # For same-rank invocation, the future may already be ready
        if hasattr(future, 'test'):
            is_ready = future.test()
            assert isinstance(is_ready, bool)

        # Get should succeed regardless
        result = future.get()
        assert result is None

    def test_invoke_async_multiple(self, context: "dtl.Context") -> None:
        """Test multiple async invocations."""
        import dtl

        def square(x):
            return x * x

        action = dtl.Action("square_async", square)

        futures = []
        for i in range(5):
            f = dtl.remote_invoke_async(context, action, context.rank, i)
            futures.append(f)

        results = [f.get() for f in futures]
        assert results == [0, 1, 4, 9, 16]


# =============================================================================
# Cross-rank remote invocation tests
# =============================================================================

@skip_no_remote
@pytest.mark.mpi
class TestRemoteInvokeCrossRank:
    """Tests for remote_invoke targeting a different rank."""

    def test_invoke_cross_rank(self, mpi_context: "dtl.Context") -> None:
        """Test invoking an action on a different rank."""
        import dtl

        def identity(x):
            return x

        action = dtl.Action("identity_cross", identity)

        target = (mpi_context.rank + 1) % mpi_context.size
        result = dtl.remote_invoke(mpi_context, action, target, 42)
        assert result == 42

    def test_invoke_async_cross_rank(self, mpi_context: "dtl.Context") -> None:
        """Test async invocation on a different rank."""
        import dtl

        def add(a, b):
            return a + b

        action = dtl.Action("add_cross", add)

        target = (mpi_context.rank + 1) % mpi_context.size
        future = dtl.remote_invoke_async(mpi_context, action, target, 10, 20)
        result = future.get()
        assert result == 30

    def test_invoke_cross_rank_round_robin(self, mpi_context: "dtl.Context") -> None:
        """Test round-robin invocation across all ranks."""
        import dtl

        def get_rank_info():
            return mpi_context.rank

        action = dtl.Action("rank_info_cross", get_rank_info)

        target = (mpi_context.rank + 1) % mpi_context.size
        result = dtl.remote_invoke(mpi_context, action, target)
        assert isinstance(result, int)
