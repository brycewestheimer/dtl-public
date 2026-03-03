# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for DTL Python bindings: probe/iprobe, gatherv, scatterv (Phase 08 parity).

These tests require MPI with at least 2 ranks for p2p probe tests.
Collective variable-count tests can run single-rank.
"""

import pytest
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dtl


# =============================================================================
# Export presence tests (always run)
# =============================================================================


class TestExports:
    """Test that new functions are exported from the dtl module."""

    def test_probe_exported(self) -> None:
        import dtl
        assert hasattr(dtl, "probe")
        assert callable(dtl.probe)

    def test_iprobe_exported(self) -> None:
        import dtl
        assert hasattr(dtl, "iprobe")
        assert callable(dtl.iprobe)

    def test_gatherv_exported(self) -> None:
        import dtl
        assert hasattr(dtl, "gatherv")
        assert callable(dtl.gatherv)

    def test_scatterv_exported(self) -> None:
        import dtl
        assert hasattr(dtl, "scatterv")
        assert callable(dtl.scatterv)


# =============================================================================
# gatherv / scatterv tests (single-rank safe)
# =============================================================================


class TestGatherv:
    """Tests for variable-count gather."""

    def test_gatherv_single_rank(self, context: "dtl.Context") -> None:
        """Test gatherv with a single rank (all data stays local)."""
        import dtl

        data = np.array([1.0, 2.0, 3.0])
        result = dtl.gatherv(context, data, root=0)

        if context.rank == 0:
            np.testing.assert_array_equal(result, data)

    def test_gatherv_with_explicit_counts(self, context: "dtl.Context") -> None:
        """Test gatherv with explicitly provided recvcounts."""
        import dtl

        data = np.array([10.0, 20.0])
        recvcounts = np.array([2] * context.size, dtype=np.int64)
        result = dtl.gatherv(context, data, recvcounts=recvcounts, root=0)

        if context.rank == 0:
            assert result.size == 2 * context.size


class TestScatterv:
    """Tests for variable-count scatter."""

    def test_scatterv_single_rank(self, context: "dtl.Context") -> None:
        """Test scatterv with a single rank."""
        import dtl

        data = np.array([1.0, 2.0, 3.0])
        sendcounts = np.array([3], dtype=np.int64)
        result = dtl.scatterv(context, data, sendcounts, root=0)

        np.testing.assert_array_equal(result, data)


# =============================================================================
# probe / iprobe tests (require MPI, 2+ ranks)
# =============================================================================


@pytest.mark.mpi
class TestProbe:
    """Tests for message probing (require 2+ MPI ranks)."""

    def test_probe_returns_dict(self, mpi_context: "dtl.Context") -> None:
        """Test that probe returns a dict with expected keys."""
        import dtl

        ctx = mpi_context
        if ctx.size < 2:
            pytest.skip("Need at least 2 ranks")

        if ctx.rank == 0:
            dtl.send(ctx, np.array([1.0, 2.0, 3.0]), dest=1, tag=42)
        elif ctx.rank == 1:
            info = dtl.probe(ctx, source=0, tag=42)
            assert isinstance(info, dict)
            assert "source" in info
            assert "tag" in info
            assert "count" in info
            assert info["source"] == 0
            assert info["tag"] == 42
            # Consume the message
            dtl.recv(ctx, count=info["count"], dtype=np.float64, source=0, tag=42)

    def test_iprobe_no_message(self, mpi_context: "dtl.Context") -> None:
        """Test iprobe when no message is pending."""
        import dtl

        ctx = mpi_context
        ctx.barrier()  # Ensure no pending messages

        found, info = dtl.iprobe(ctx, source=-2, tag=-1)
        # May or may not find a message; just check the return type
        assert isinstance(found, bool)
        if found:
            assert isinstance(info, dict)
        else:
            assert info is None
