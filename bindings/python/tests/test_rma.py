# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
DTL Python Bindings - RMA Tests

Tests for Remote Memory Access (RMA) one-sided communication.
"""

import pytest
import numpy as np
import dtl


class TestWindowCreation:
    """Tests for Window creation and basic operations."""

    def test_window_allocate(self):
        """Test creating a window with allocated memory."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            assert win.size == 1024
            assert win.is_valid

    def test_window_from_array(self):
        """Test creating a window from a NumPy array."""
        with dtl.Context() as ctx:
            data = np.zeros(100, dtype=np.float64)
            win = dtl.Window(ctx, base=data)
            assert win.size == data.nbytes
            assert win.is_valid

    def test_window_with_explicit_size(self):
        """Test creating a window with explicit size."""
        with dtl.Context() as ctx:
            data = np.zeros(100, dtype=np.float64)
            win = dtl.Window(ctx, size=400, base=data)  # Only expose 400 bytes
            assert win.size == 400
            assert win.is_valid

    def test_window_base_address(self):
        """Test that base address is accessible."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            base = win.base
            # Base should be an integer address or None
            assert base is None or isinstance(base, int)


class TestWindowSynchronization:
    """Tests for window synchronization operations."""

    def test_fence_basic(self):
        """Test basic fence synchronization."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            # Fence is collective, should work even with single rank
            win.fence()
            win.fence()  # Multiple fences should work

    def test_lock_unlock_self(self):
        """Test locking own window."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            # Lock own rank
            win.lock(target=ctx.rank)
            win.unlock(target=ctx.rank)

    def test_lock_exclusive(self):
        """Test exclusive lock mode."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock(target=ctx.rank, mode="exclusive")
            win.unlock(target=ctx.rank)

    def test_lock_shared(self):
        """Test shared lock mode."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock(target=ctx.rank, mode="shared")
            win.unlock(target=ctx.rank)

    def test_lock_all_unlock_all(self):
        """Test lock_all and unlock_all."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock_all()
            win.unlock_all()

    def test_flush(self):
        """Test flush operation."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock(target=ctx.rank)
            win.flush(target=ctx.rank)
            win.unlock(target=ctx.rank)

    def test_flush_all(self):
        """Test flush_all operation."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock_all()
            win.flush_all()
            win.unlock_all()

    def test_flush_local(self):
        """Test flush_local operation."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock(target=ctx.rank)
            win.flush_local(target=ctx.rank)
            win.unlock(target=ctx.rank)

    def test_flush_local_all(self):
        """Test flush_local_all operation."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            win.lock_all()
            win.flush_local_all()
            win.unlock_all()


class TestRMAPutGet:
    """Tests for RMA put and get operations."""

    def test_put_to_self(self):
        """Test putting data to own window."""
        with dtl.Context() as ctx:
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            win = dtl.Window(ctx, size=data.nbytes)

            win.fence()
            dtl.rma_put(win, target=ctx.rank, offset=0, data=data)
            win.fence()

    def test_get_from_self(self):
        """Test getting data from own window."""
        with dtl.Context() as ctx:
            data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            win = dtl.Window(ctx, base=data)

            win.fence()
            result = dtl.rma_get(win, target=ctx.rank, offset=0,
                                 size=data.nbytes, dtype=np.float64)
            win.fence()

            np.testing.assert_array_equal(result, data)

    def test_put_get_roundtrip(self):
        """Test putting then getting data."""
        with dtl.Context() as ctx:
            source = np.array([10.0, 20.0, 30.0], dtype=np.float64)
            buffer = np.zeros(3, dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            win.fence()
            dtl.rma_put(win, target=ctx.rank, offset=0, data=source)
            win.fence()

            result = dtl.rma_get(win, target=ctx.rank, offset=0,
                                 size=source.nbytes, dtype=np.float64)
            win.fence()

            np.testing.assert_array_equal(result, source)

    def test_put_with_offset(self):
        """Test putting data at non-zero offset."""
        with dtl.Context() as ctx:
            buffer = np.zeros(10, dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            data = np.array([1.0, 2.0], dtype=np.float64)
            offset = 4 * 8  # Skip 4 doubles (32 bytes)

            win.fence()
            dtl.rma_put(win, target=ctx.rank, offset=offset, data=data)
            win.fence()

            # Read back
            result = dtl.rma_get(win, target=ctx.rank, offset=offset,
                                 size=data.nbytes, dtype=np.float64)
            win.fence()

            np.testing.assert_array_equal(result, data)

    def test_get_partial(self):
        """Test getting partial data."""
        with dtl.Context() as ctx:
            data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
            win = dtl.Window(ctx, base=data)

            win.fence()
            # Get only 2 elements starting at index 1
            result = dtl.rma_get(win, target=ctx.rank, offset=8,
                                 size=16, dtype=np.float64)
            win.fence()

            np.testing.assert_array_equal(result, [2.0, 3.0])


class TestRMAAccumulate:
    """Tests for RMA accumulate operations."""

    def test_accumulate_sum(self):
        """Test accumulate with sum operation."""
        with dtl.Context() as ctx:
            buffer = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            addend = np.array([10.0, 20.0, 30.0], dtype=np.float64)

            win.fence()
            dtl.rma_accumulate(win, target=ctx.rank, offset=0, data=addend, op="sum")
            win.fence()

            # Buffer should now be [11.0, 22.0, 33.0]
            np.testing.assert_array_equal(buffer, [11.0, 22.0, 33.0])

    def test_accumulate_prod(self):
        """Test accumulate with product operation."""
        with dtl.Context() as ctx:
            buffer = np.array([2.0, 3.0, 4.0], dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            multiplier = np.array([2.0, 2.0, 2.0], dtype=np.float64)

            win.fence()
            dtl.rma_accumulate(win, target=ctx.rank, offset=0, data=multiplier, op="prod")
            win.fence()

            np.testing.assert_array_equal(buffer, [4.0, 6.0, 8.0])

    def test_accumulate_min(self):
        """Test accumulate with min operation."""
        with dtl.Context() as ctx:
            buffer = np.array([5.0, 10.0, 15.0], dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            values = np.array([3.0, 20.0, 10.0], dtype=np.float64)

            win.fence()
            dtl.rma_accumulate(win, target=ctx.rank, offset=0, data=values, op="min")
            win.fence()

            np.testing.assert_array_equal(buffer, [3.0, 10.0, 10.0])

    def test_accumulate_max(self):
        """Test accumulate with max operation."""
        with dtl.Context() as ctx:
            buffer = np.array([5.0, 10.0, 15.0], dtype=np.float64)
            win = dtl.Window(ctx, base=buffer)

            values = np.array([3.0, 20.0, 10.0], dtype=np.float64)

            win.fence()
            dtl.rma_accumulate(win, target=ctx.rank, offset=0, data=values, op="max")
            win.fence()

            np.testing.assert_array_equal(buffer, [5.0, 20.0, 15.0])


class TestRMAAtomics:
    """Tests for RMA atomic operations."""

    def test_fetch_and_add(self):
        """Test atomic fetch and add."""
        with dtl.Context() as ctx:
            buffer = np.array([100], dtype=np.int64)
            win = dtl.Window(ctx, base=buffer)

            win.lock(target=ctx.rank)
            old = dtl.rma_fetch_and_add(win, target=ctx.rank, offset=0,
                                        addend=5, dtype=np.int64)
            win.flush(target=ctx.rank)
            win.unlock(target=ctx.rank)

            assert old == 100
            assert buffer[0] == 105

    def test_compare_and_swap_success(self):
        """Test compare and swap when compare matches."""
        with dtl.Context() as ctx:
            buffer = np.array([42], dtype=np.int64)
            win = dtl.Window(ctx, base=buffer)

            win.lock(target=ctx.rank)
            old = dtl.rma_compare_and_swap(win, target=ctx.rank, offset=0,
                                           compare=42, swap=100, dtype=np.int64)
            win.flush(target=ctx.rank)
            win.unlock(target=ctx.rank)

            assert old == 42  # Should return original value
            assert buffer[0] == 100  # Should be swapped

    def test_compare_and_swap_failure(self):
        """Test compare and swap when compare doesn't match."""
        with dtl.Context() as ctx:
            buffer = np.array([42], dtype=np.int64)
            win = dtl.Window(ctx, base=buffer)

            win.lock(target=ctx.rank)
            old = dtl.rma_compare_and_swap(win, target=ctx.rank, offset=0,
                                           compare=99, swap=100, dtype=np.int64)
            win.flush(target=ctx.rank)
            win.unlock(target=ctx.rank)

            assert old == 42  # Should return original value
            assert buffer[0] == 42  # Should NOT be swapped


class TestRMADtypes:
    """Tests for RMA operations with different data types."""

    @pytest.mark.parametrize("dtype", [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float32, np.float64
    ])
    def test_put_get_dtype(self, dtype):
        """Test put/get with various dtypes."""
        with dtl.Context() as ctx:
            data = np.array([1, 2, 3, 4], dtype=dtype)
            win = dtl.Window(ctx, size=data.nbytes)

            win.fence()
            dtl.rma_put(win, target=ctx.rank, offset=0, data=data)
            win.fence()

            result = dtl.rma_get(win, target=ctx.rank, offset=0,
                                 size=data.nbytes, dtype=dtype)
            win.fence()

            np.testing.assert_array_equal(result, data)


class TestWindowContextManager:
    """Tests for window context manager protocol."""

    def test_context_manager(self):
        """Test window as context manager."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            with win:
                # Inside context manager
                assert win.is_valid
            # Window should still be valid after __exit__
            # (we don't auto-destroy on exit per current design)


class TestRMAErrors:
    """Tests for RMA error handling."""

    def test_invalid_target_rank(self):
        """Test error on invalid target rank."""
        with dtl.Context() as ctx:
            win = dtl.Window(ctx, size=1024)
            data = np.array([1.0, 2.0], dtype=np.float64)

            win.fence()
            # Target rank out of range should fail
            with pytest.raises(RuntimeError):
                dtl.rma_put(win, target=ctx.size + 10, offset=0, data=data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
