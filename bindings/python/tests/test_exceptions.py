# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DTL custom exception hierarchy.

Verifies that the custom exception classes defined in dtl/__init__.py
are properly registered and raised by the C++ status_exception.hpp
mapping instead of builtin Python exceptions.
"""

import pytest
import dtl


class TestExceptionHierarchy:
    """Verify the exception class hierarchy is correct."""

    def test_dtl_error_is_base(self):
        assert issubclass(dtl.CommunicationError, dtl.DTLError)
        assert issubclass(dtl.MemoryError, dtl.DTLError)
        assert issubclass(dtl.BoundsError, dtl.DTLError)
        assert issubclass(dtl.InvalidArgumentError, dtl.DTLError)
        assert issubclass(dtl.BackendError, dtl.DTLError)

    def test_bounds_error_is_index_error(self):
        assert issubclass(dtl.BoundsError, IndexError)

    def test_invalid_argument_error_is_value_error(self):
        assert issubclass(dtl.InvalidArgumentError, ValueError)

    def test_memory_error_is_builtin_memory_error(self):
        assert issubclass(dtl.MemoryError, MemoryError)

    def test_dtl_error_is_exception(self):
        assert issubclass(dtl.DTLError, Exception)


class TestExceptionInstantiation:
    """Verify custom exceptions can be raised and caught."""

    def test_catch_dtl_error(self):
        with pytest.raises(dtl.DTLError):
            raise dtl.DTLError("test")

    def test_catch_communication_error(self):
        with pytest.raises(dtl.CommunicationError):
            raise dtl.CommunicationError("test")

    def test_catch_communication_error_as_dtl_error(self):
        with pytest.raises(dtl.DTLError):
            raise dtl.CommunicationError("test")

    def test_catch_bounds_error_as_index_error(self):
        with pytest.raises(IndexError):
            raise dtl.BoundsError("test")

    def test_catch_invalid_argument_as_value_error(self):
        with pytest.raises(ValueError):
            raise dtl.InvalidArgumentError("test")

    def test_catch_backend_error(self):
        with pytest.raises(dtl.BackendError):
            raise dtl.BackendError("test")
