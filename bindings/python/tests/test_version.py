# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings version information (Phase 28).

Tests cover:
- __version__ string format
- version_info tuple format and values
- Consistency between __version__ and version_info
"""

import re


class TestVersion:
    """Tests for version information."""

    def test_version_string_exists(self) -> None:
        """Test that __version__ is defined."""
        import dtl
        assert hasattr(dtl, "__version__")
        assert isinstance(dtl.__version__, str)

    def test_version_format(self) -> None:
        """Test that __version__ follows PEP 440 format (with optional alpha)."""
        import dtl
        pattern = r"^\d+\.\d+\.\d+(a\d+)?$"
        assert re.match(pattern, dtl.__version__), (
            f"Version '{dtl.__version__}' does not match format X.Y.Z or X.Y.ZaN"
        )

    def test_version_components_are_numeric(self) -> None:
        """Test that __version__ contains numeric semver base components."""
        import dtl
        # Strip PEP 440 prerelease suffix before checking
        base = re.sub(r"(a|b|rc)\d+$", "", dtl.__version__)
        parts = base.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)

    def test_version_info_exists(self) -> None:
        """Test that version_info tuple is defined."""
        import dtl
        assert hasattr(dtl, "version_info")

    def test_version_info_is_tuple(self) -> None:
        """Test that version_info is a tuple."""
        import dtl
        assert isinstance(dtl.version_info, tuple)

    def test_version_info_length(self) -> None:
        """Test that version_info has 3 elements."""
        import dtl
        assert len(dtl.version_info) == 3

    def test_version_info_values(self) -> None:
        """Test that version_info matches expected base version."""
        import dtl
        assert dtl.version_info == (0, 1, 0)

    def test_version_info_matches_string(self) -> None:
        """Test that version_info matches the base part of __version__."""
        import dtl
        base = ".".join(str(x) for x in dtl.version_info)
        assert dtl.__version__.startswith(base)

    def test_version_info_elements_are_ints(self) -> None:
        """Test that all version_info elements are integers."""
        import dtl
        for elem in dtl.version_info:
            assert isinstance(elem, int)
