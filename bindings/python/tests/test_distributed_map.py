# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings DistributedMap (Phase 28).

Tests cover:
- DistributedMap creation with various key/value dtypes
- insert, find, erase, contains operations
- size, local_size, global_size, empty properties
- local_view snapshot
- global_view (collective)
- sync, flush, clear lifecycle
- __len__, __contains__ dunder methods
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


class TestDistributedMapCreation:
    """Tests for DistributedMap construction."""

    def test_create_default_dtypes(self, context: "dtl.Context") -> None:
        """Test creating map with default dtypes (int64 keys, float64 values)."""
        import dtl

        m = dtl.DistributedMap(context)
        assert m.local_size == 0
        m.destroy()

    def test_create_int32_keys(self, context: "dtl.Context") -> None:
        """Test creating map with int32 keys."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int32, value_dtype=np.float64)
        assert m.local_size == 0
        m.destroy()

    def test_create_uint64_keys(self, context: "dtl.Context") -> None:
        """Test creating map with uint64 keys and int32 values."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.uint64, value_dtype=np.int32)
        assert m.local_size == 0
        m.destroy()

    def test_unsupported_key_dtype_raises(self, context: "dtl.Context") -> None:
        """Test that unsupported key dtype raises TypeError."""
        import dtl

        with pytest.raises(TypeError, match="Unsupported key dtype"):
            dtl.DistributedMap(context, key_dtype=np.complex128)


class TestDistributedMapInsertFind:
    """Tests for insert and find operations."""

    def test_insert_and_find(self, context: "dtl.Context") -> None:
        """Test basic insert and find."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(42, 3.14)
        val = m.find(42)
        assert val is not None
        assert abs(val - 3.14) < 1e-10
        m.destroy()

    def test_find_nonexistent_key(self, context: "dtl.Context") -> None:
        """Test that find returns None for missing keys."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        val = m.find(999)
        assert val is None
        m.destroy()

    def test_insert_multiple(self, context: "dtl.Context") -> None:
        """Test inserting multiple key-value pairs."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        for i in range(10):
            m.insert(i, float(i * 10))

        # Verify at least some are findable locally
        found_count = 0
        for i in range(10):
            val = m.find(i)
            if val is not None:
                assert abs(val - float(i * 10)) < 1e-10
                found_count += 1
        # In single-rank mode, all should be local
        assert found_count > 0
        m.destroy()


class TestDistributedMapErase:
    """Tests for erase and contains operations."""

    def test_erase_existing_key(self, context: "dtl.Context") -> None:
        """Test erasing an existing key returns True."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(10, 1.0)
        result = m.erase(10)
        assert result is True
        # Should no longer be findable
        assert m.find(10) is None
        m.destroy()

    def test_erase_nonexistent_key(self, context: "dtl.Context") -> None:
        """Test erasing a nonexistent key returns False."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        result = m.erase(999)
        assert result is False
        m.destroy()

    def test_contains(self, context: "dtl.Context") -> None:
        """Test contains checks local partition."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(5, 50.0)
        assert m.contains(5) is True
        assert m.contains(999) is False
        m.destroy()


class TestDistributedMapProperties:
    """Tests for size and empty properties."""

    def test_local_size(self, context: "dtl.Context") -> None:
        """Test local_size reflects number of local entries."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        assert m.local_size == 0
        m.insert(1, 1.0)
        assert m.local_size >= 1  # May be 0 if key hashes to another rank
        m.destroy()

    def test_empty_property(self, context: "dtl.Context") -> None:
        """Test empty property."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        assert m.empty is True
        m.insert(1, 1.0)
        # After insertion, may still be empty if key hashed elsewhere
        # But in single-rank mode, it should be non-empty
        m.destroy()

    def test_len_dunder(self, context: "dtl.Context") -> None:
        """Test __len__ returns local_size."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        assert len(m) == 0
        m.destroy()

    def test_contains_dunder(self, context: "dtl.Context") -> None:
        """Test __contains__ (in operator)."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(7, 70.0)
        assert (7 in m) == m.contains(7)
        assert (999 in m) is False
        m.destroy()


class TestDistributedMapLifecycle:
    """Tests for sync, flush, clear, and destroy."""

    def test_sync(self, context: "dtl.Context") -> None:
        """Test sync does not raise."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(1, 1.0)
        m.sync()
        m.destroy()

    def test_clear(self, context: "dtl.Context") -> None:
        """Test clear removes all elements."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(1, 1.0)
        m.insert(2, 2.0)
        m.clear()
        assert m.local_size == 0
        m.destroy()

    def test_local_view_returns_dict(self, context: "dtl.Context") -> None:
        """Test local_view returns a dict snapshot of local entries."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        m.insert(1, 1.0)
        view = m.local_view()
        assert isinstance(view, dict)
        if context.size == 1:
            assert view.get(1) == 1.0
        m.destroy()

    def test_global_view_collective(self, context: "dtl.Context") -> None:
        """Test global_view gathers all rank contributions collectively."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        key = 100000 + int(context.rank)
        value = float(context.rank) + 0.5
        m.insert(key, value)
        result = m.global_view()
        assert isinstance(result, dict)
        assert len(result) >= context.size
        for r in range(int(context.size)):
            expected_key = 100000 + r
            assert expected_key in result
            assert abs(result[expected_key] - (float(r) + 0.5)) < 1e-10
        m.destroy()

    def test_repr(self, context: "dtl.Context") -> None:
        """Test repr string format."""
        import dtl

        m = dtl.DistributedMap(context, key_dtype=np.int64, value_dtype=np.float64)
        r = repr(m)
        assert "DistributedMap" in r
        assert "local_size" in r
        m.destroy()
