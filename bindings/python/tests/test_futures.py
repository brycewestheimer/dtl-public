# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings futures module (experimental).

Tests cover:
- Future creation and destruction
- Set/get value roundtrip
- Wait on completed future
- test() readiness check after set
- when_all with multiple futures
- when_any with multiple futures

Note: The futures module is experimental and may change in future releases.
"""

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# Check if futures bindings are available
try:
    import dtl
    HAS_FUTURES = hasattr(dtl, 'Future')
except ImportError:
    HAS_FUTURES = False

skip_no_futures = pytest.mark.skipif(
    not HAS_FUTURES,
    reason="Futures bindings not available (experimental)"
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@skip_no_futures
class TestFutureCreation:
    """Tests for Future creation and destruction."""

    def test_create_future(self) -> None:
        """Test creating a Future object."""
        import dtl

        future = dtl.Future()
        assert future is not None

    def test_create_and_destroy(self) -> None:
        """Test that Future can be created and garbage collected."""
        import dtl

        future = dtl.Future()
        assert future is not None
        del future  # Should not raise

    def test_future_not_ready_initially(self) -> None:
        """Test that a new Future is not ready."""
        import dtl

        future = dtl.Future()
        assert future.test() is False

    def test_future_valid(self) -> None:
        """Test that a newly created Future is valid."""
        import dtl

        future = dtl.Future()
        assert future.valid() is True


@skip_no_futures
class TestFutureSetGet:
    """Tests for Future set/get value operations."""

    def test_set_and_get_int(self) -> None:
        """Test setting and getting an integer value."""
        import dtl

        future = dtl.Future()
        future.set(42)

        result = future.get()
        assert result == 42

    def test_set_and_get_float(self) -> None:
        """Test setting and getting a float value."""
        import dtl

        future = dtl.Future()
        future.set(3.14)

        result = future.get()
        assert abs(result - 3.14) < 1e-10

    def test_set_and_get_string(self) -> None:
        """Test setting and getting a string value."""
        import dtl

        future = dtl.Future()
        future.set("hello")

        result = future.get()
        assert result == "hello"

    def test_set_and_get_none(self) -> None:
        """Test setting and getting None."""
        import dtl

        future = dtl.Future()
        future.set(None)

        result = future.get()
        assert result is None

    def test_get_after_set_returns_value(self) -> None:
        """Test that get returns the value that was set."""
        import dtl

        future = dtl.Future()
        future.set(99)

        assert future.get() == 99


@skip_no_futures
class TestFutureWait:
    """Tests for Future wait operations."""

    def test_wait_on_completed_future(self) -> None:
        """Test that wait returns immediately on a completed future."""
        import dtl

        future = dtl.Future()
        future.set(42)

        # wait() should return immediately since value is already set
        future.wait()

        # Value should still be accessible
        assert future.get() == 42

    def test_wait_does_not_consume_value(self) -> None:
        """Test that wait does not consume the future's value."""
        import dtl

        future = dtl.Future()
        future.set(100)

        future.wait()
        future.wait()  # Multiple waits should be fine

        assert future.get() == 100


@skip_no_futures
class TestFutureTest:
    """Tests for Future readiness check."""

    def test_not_ready_before_set(self) -> None:
        """Test that test() returns False before set."""
        import dtl

        future = dtl.Future()
        assert future.test() is False

    def test_ready_after_set(self) -> None:
        """Test that test() returns True after set."""
        import dtl

        future = dtl.Future()
        future.set(42)

        assert future.test() is True

    def test_test_returns_bool(self) -> None:
        """Test that test() returns a boolean."""
        import dtl

        future = dtl.Future()
        result = future.test()
        assert isinstance(result, bool)

    def test_test_is_non_blocking(self) -> None:
        """Test that test() returns without blocking."""
        import dtl

        future = dtl.Future()
        # This should return immediately, not block
        result = future.test()
        assert result is False


@skip_no_futures
class TestWhenAll:
    """Tests for when_all combinator."""

    def test_when_all_single_future(self) -> None:
        """Test when_all with a single future."""
        import dtl

        f1 = dtl.Future()
        f1.set(10)

        combined = dtl.when_all([f1])
        results = combined.get()
        assert results == [10]

    def test_when_all_multiple_futures(self) -> None:
        """Test when_all with multiple futures."""
        import dtl

        f1 = dtl.Future()
        f2 = dtl.Future()
        f3 = dtl.Future()

        f1.set(1)
        f2.set(2)
        f3.set(3)

        combined = dtl.when_all([f1, f2, f3])
        results = combined.get()
        assert results == [1, 2, 3]

    def test_when_all_empty_list(self) -> None:
        """Test when_all with empty list."""
        import dtl

        combined = dtl.when_all([])
        results = combined.get()
        assert results == []

    def test_when_all_is_ready_when_all_set(self) -> None:
        """Test when_all future is ready when all inputs are set."""
        import dtl

        f1 = dtl.Future()
        f2 = dtl.Future()

        f1.set(10)
        f2.set(20)

        combined = dtl.when_all([f1, f2])
        assert combined.test() is True


@skip_no_futures
class TestWhenAny:
    """Tests for when_any combinator."""

    def test_when_any_single_future(self) -> None:
        """Test when_any with a single completed future."""
        import dtl

        f1 = dtl.Future()
        f1.set(42)

        result_future = dtl.when_any([f1])
        index, value = result_future.get()
        assert index == 0
        assert value == 42

    def test_when_any_first_ready(self) -> None:
        """Test when_any returns index and value of first completed future."""
        import dtl

        f1 = dtl.Future()
        f2 = dtl.Future()
        f3 = dtl.Future()

        # Only set f2
        f2.set(99)

        result_future = dtl.when_any([f1, f2, f3])
        index, value = result_future.get()
        assert index == 1
        assert value == 99

    def test_when_any_all_ready(self) -> None:
        """Test when_any when all futures are ready."""
        import dtl

        f1 = dtl.Future()
        f2 = dtl.Future()

        f1.set(10)
        f2.set(20)

        result_future = dtl.when_any([f1, f2])
        index, value = result_future.get()

        # Should return one of the completed futures
        assert index in (0, 1)
        if index == 0:
            assert value == 10
        else:
            assert value == 20

    def test_when_any_is_ready_when_one_set(self) -> None:
        """Test when_any future is ready when at least one input is set."""
        import dtl

        f1 = dtl.Future()
        f2 = dtl.Future()

        f1.set(42)
        # f2 is not set

        result_future = dtl.when_any([f1, f2])
        assert result_future.test() is True
