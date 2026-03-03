# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

"""
DTL Python MPI Integration Tests

Tests the Python bindings under MPI, covering single-rank and multi-rank
scenarios for context management, containers, collective operations,
algorithms, and feature detection.

Run single-rank:
    python3 -m pytest test_python_mpi_integration.py -v

Run multi-rank:
    mpirun -np 4 python3 -m pytest test_python_mpi_integration.py -v
"""

import pytest
import numpy as np

# Skip all tests if dtl is not available
dtl = pytest.importorskip("dtl")


# =============================================================================
# Feature Detection
# =============================================================================


class TestFeatureDetection:
    """Test feature detection queries."""

    def test_has_mpi_returns_bool(self):
        assert isinstance(dtl.has_mpi(), bool)

    def test_has_cuda_returns_bool(self):
        assert isinstance(dtl.has_cuda(), bool)

    def test_has_hip_returns_bool(self):
        assert isinstance(dtl.has_hip(), bool)

    def test_has_nccl_returns_bool(self):
        assert isinstance(dtl.has_nccl(), bool)

    def test_has_shmem_returns_bool(self):
        assert isinstance(dtl.has_shmem(), bool)

    def test_version_string(self):
        assert isinstance(dtl.__version__, str)
        assert len(dtl.__version__) > 0

    def test_version_info(self):
        assert isinstance(dtl.version_info, tuple)
        assert len(dtl.version_info) == 3
        for component in dtl.version_info:
            assert isinstance(component, int)
            assert component >= 0


# =============================================================================
# Context Tests
# =============================================================================


class TestMpiContext:
    """Test Context creation and properties under MPI."""

    def test_context_creation(self):
        with dtl.Context() as ctx:
            assert ctx.rank >= 0
            assert ctx.size >= 1

    def test_context_rank_within_bounds(self):
        with dtl.Context() as ctx:
            assert 0 <= ctx.rank < ctx.size

    def test_context_is_root(self):
        with dtl.Context() as ctx:
            assert ctx.is_root == (ctx.rank == 0)

    def test_context_device_id_default(self):
        with dtl.Context() as ctx:
            assert ctx.device_id == -1
            assert not ctx.has_device

    def test_context_barrier(self):
        with dtl.Context() as ctx:
            # Should complete without error
            ctx.barrier()

    def test_context_multiple_barriers(self):
        with dtl.Context() as ctx:
            for _ in range(5):
                ctx.barrier()

    def test_context_repr(self):
        with dtl.Context() as ctx:
            r = repr(ctx)
            assert "Context" in r
            assert "rank=" in r
            assert "size=" in r

    @pytest.mark.mpi
    def test_context_with_mpi_comm(self):
        from mpi4py import MPI
        with dtl.Context(comm=MPI.COMM_WORLD) as ctx:
            assert ctx.rank == MPI.COMM_WORLD.Get_rank()
            assert ctx.size == MPI.COMM_WORLD.Get_size()

    @pytest.mark.mpi
    def test_context_with_none_comm(self):
        ctx = dtl.Context(comm=None)
        assert ctx is not None
        assert ctx.rank >= 0
        assert ctx.size >= 1


class TestMultiRankContext:
    """Tests that specifically validate multi-rank MPI behavior."""

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_multi_rank_size(self, mpi_context):
        """Verify we actually have multiple ranks."""
        assert mpi_context.size >= 2

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_unique_ranks(self, mpi_context):
        """Each rank should have a unique rank ID."""
        from mpi4py import MPI
        all_ranks = MPI.COMM_WORLD.allgather(mpi_context.rank)
        assert len(set(all_ranks)) == mpi_context.size

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_barrier_synchronizes(self, mpi_context):
        """Barrier should synchronize all ranks."""
        import time
        # Stagger start times by rank
        time.sleep(mpi_context.rank * 0.01)
        mpi_context.barrier()
        # All ranks should reach here


# =============================================================================
# Container Tests
# =============================================================================


class TestDistributedVector:
    """Test DistributedVector creation and operations under MPI."""

    def test_create_float64(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        assert vec.global_size == 100
        assert vec.local_size > 0
        assert vec.local_size <= 100

    def test_create_float32(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float32)
        assert vec.global_size == 100

    def test_create_int64(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.int64)
        assert vec.global_size == 100

    def test_create_int32(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.int32)
        assert vec.global_size == 100

    def test_create_default_dtype(self, context):
        vec = dtl.DistributedVector(context, size=100)
        local = vec.local_view()
        assert local.dtype == np.float64

    def test_create_with_fill(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=42.0)
        local = vec.local_view()
        assert np.all(local == 42.0)

    def test_global_size(self, context):
        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert vec.global_size == 1000

    def test_local_offset(self, context):
        vec = dtl.DistributedVector(context, size=1000, dtype=np.float64)
        assert vec.local_offset >= 0
        assert vec.local_offset < vec.global_size

    def test_len(self, context):
        vec = dtl.DistributedVector(context, size=500, dtype=np.float64)
        assert len(vec) == 500

    def test_local_view_returns_numpy(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        assert isinstance(local, np.ndarray)

    def test_local_view_correct_dtype(self, context):
        for dtype in [np.float64, np.float32, np.int64, np.int32]:
            vec = dtl.DistributedVector(context, size=100, dtype=dtype)
            local = vec.local_view()
            assert local.dtype == dtype

    def test_local_view_correct_size(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        assert len(local) == vec.local_size

    def test_local_view_is_writable(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local = vec.local_view()
        local[:] = 123.0
        local2 = vec.local_view()
        assert np.all(local2 == 123.0)

    def test_local_view_shares_memory(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        local1 = vec.local_view()
        local2 = vec.local_view()
        local1[0] = 999.0
        assert local2[0] == 999.0

    def test_fill(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        vec.fill(42.5)
        local = vec.local_view()
        assert np.all(local == 42.5)

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_distributed_partitioning(self, mpi_context):
        """In multi-rank, local sizes should sum to global size."""
        from mpi4py import MPI
        size = 1000
        vec = dtl.DistributedVector(mpi_context, size=size, dtype=np.float64)
        total_local = MPI.COMM_WORLD.allreduce(vec.local_size, op=MPI.SUM)
        assert total_local == size

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_non_overlapping_offsets(self, mpi_context):
        """Local offsets should not overlap across ranks."""
        from mpi4py import MPI
        vec = dtl.DistributedVector(mpi_context, size=1000, dtype=np.float64)
        all_offsets = MPI.COMM_WORLD.allgather(vec.local_offset)
        all_sizes = MPI.COMM_WORLD.allgather(vec.local_size)

        # Check no overlapping ranges
        ranges = sorted(zip(all_offsets, all_sizes))
        for i in range(1, len(ranges)):
            prev_end = ranges[i - 1][0] + ranges[i - 1][1]
            assert prev_end <= ranges[i][0]


# =============================================================================
# Collective Operation Tests
# =============================================================================


class TestAllreduce:
    """Test allreduce collective operation."""

    def test_allreduce_sum_scalar(self, context):
        result = dtl.allreduce(context, 1.0, op="sum")
        assert result == float(context.size)

    def test_allreduce_sum_array(self, context):
        data = np.ones(10, dtype=np.float64)
        result = dtl.allreduce(context, data, op="sum")
        expected = np.full(10, float(context.size), dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_allreduce_prod(self, context):
        data = np.full(5, 2.0, dtype=np.float64)
        result = dtl.allreduce(context, data, op="prod")
        expected = np.full(5, 2.0 ** context.size, dtype=np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_allreduce_min(self, context):
        data = np.array([float(context.rank)], dtype=np.float64)
        result = dtl.allreduce(context, data, op="min")
        assert result[0] == 0.0

    def test_allreduce_max(self, context):
        data = np.array([float(context.rank)], dtype=np.float64)
        result = dtl.allreduce(context, data, op="max")
        assert result[0] == float(context.size - 1)

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_allreduce_multirank_sum(self, mpi_context):
        """Each rank contributes its rank value; sum = 0+1+...+(n-1)."""
        data = np.array([float(mpi_context.rank)], dtype=np.float64)
        result = dtl.allreduce(mpi_context, data, op="sum")
        n = mpi_context.size
        expected = n * (n - 1) / 2.0
        assert abs(result[0] - expected) < 1e-10


class TestReduce:
    """Test reduce collective operation."""

    def test_reduce_sum_to_root(self, context):
        data = np.ones(10, dtype=np.float64)
        result = dtl.reduce(context, data, op="sum", root=0)
        if context.rank == 0:
            expected = np.full(10, float(context.size), dtype=np.float64)
            np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_reduce_multirank(self, mpi_context):
        """Reduce rank values to root."""
        data = np.array([float(mpi_context.rank)], dtype=np.float64)
        result = dtl.reduce(mpi_context, data, op="sum", root=0)
        if mpi_context.rank == 0:
            n = mpi_context.size
            expected = n * (n - 1) / 2.0
            assert abs(result[0] - expected) < 1e-10


class TestBroadcast:
    """Test broadcast collective operation."""

    def test_broadcast_from_root(self, context):
        if context.rank == 0:
            data = np.array([42.0, 43.0, 44.0], dtype=np.float64)
        else:
            data = np.zeros(3, dtype=np.float64)
        result = dtl.broadcast(context, data, root=0)
        np.testing.assert_array_equal(result, np.array([42.0, 43.0, 44.0]))

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_broadcast_multirank(self, mpi_context):
        """Root broadcasts data, all ranks receive same values."""
        if mpi_context.rank == 0:
            data = np.arange(5, dtype=np.float64) * 10.0
        else:
            data = np.zeros(5, dtype=np.float64)
        result = dtl.broadcast(mpi_context, data, root=0)
        expected = np.arange(5, dtype=np.float64) * 10.0
        np.testing.assert_array_equal(result, expected)


class TestGather:
    """Test gather collective operation."""

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_gather_to_root(self, mpi_context):
        """Each rank sends its rank value; root receives all."""
        data = np.array([float(mpi_context.rank)], dtype=np.float64)
        result = dtl.gather(mpi_context, data, root=0)
        if mpi_context.rank == 0:
            assert len(result) == mpi_context.size
            for i in range(mpi_context.size):
                assert result[i] == float(i)


class TestScatter:
    """Test scatter collective operation."""

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_scatter_from_root(self, mpi_context):
        """Root scatters an array; each rank gets its chunk."""
        if mpi_context.rank == 0:
            data = np.arange(mpi_context.size, dtype=np.float64) * 100.0
        else:
            data = np.zeros(mpi_context.size, dtype=np.float64)
        result = dtl.scatter(mpi_context, data, root=0)
        assert result[0] == float(mpi_context.rank) * 100.0


class TestAllgather:
    """Test allgather collective operation."""

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_allgather(self, mpi_context):
        """Each rank sends rank value; all receive all values."""
        data = np.array([float(mpi_context.rank)], dtype=np.float64)
        result = dtl.allgather(mpi_context, data)
        assert len(result) == mpi_context.size
        for i in range(mpi_context.size):
            assert result[i] == float(i)


class TestSendRecv:
    """Test point-to-point operations."""

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_send_recv(self, mpi_context):
        """Rank 0 sends to rank 1, rank 1 receives."""
        if mpi_context.size < 2:
            pytest.skip("Need at least 2 ranks")
        if mpi_context.rank == 0:
            data = np.array([3.14, 2.72], dtype=np.float64)
            dtl.send(mpi_context, data, dest=1, tag=42)
        elif mpi_context.rank == 1:
            result = dtl.recv(mpi_context, count=2, dtype=np.float64,
                              source=0, tag=42)
            np.testing.assert_array_almost_equal(
                result, np.array([3.14, 2.72]))
        mpi_context.barrier()

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_sendrecv(self, mpi_context):
        """Each rank exchanges data with next rank in a ring."""
        if mpi_context.size < 2:
            pytest.skip("Need at least 2 ranks")
        dest = (mpi_context.rank + 1) % mpi_context.size
        source = (mpi_context.rank - 1) % mpi_context.size
        send_data = np.array([float(mpi_context.rank)], dtype=np.float64)
        result = dtl.sendrecv(mpi_context, send_data, dest=dest,
                              recvcount=1, recvdtype=np.float64, source=source)
        assert result[0] == float(source)


# =============================================================================
# Algorithm Tests
# =============================================================================


class TestAlgorithms:
    """Test DTL algorithms under MPI."""

    def test_fill_vector(self, context):
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64)
        dtl.fill_vector(vec, 99.0)
        local = vec.local_view()
        assert np.all(local == 99.0)

    def test_copy_vector(self, context):
        src = dtl.DistributedVector(context, size=50, dtype=np.float64, fill=7.0)
        dst = dtl.DistributedVector(context, size=50, dtype=np.float64)
        dtl.copy_vector(src, dst)
        np.testing.assert_array_equal(dst.local_view(), src.local_view())

    def test_for_each_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=5.0)
        values = []
        dtl.for_each_vector(vec, lambda x: values.append(x))
        assert len(values) == vec.local_size
        assert all(v == 5.0 for v in values)

    def test_transform_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=3.0)
        dtl.transform_vector(vec, lambda x: x * 2)
        local = vec.local_view()
        assert np.all(local == 6.0)

    def test_count_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.int32, fill=5)
        count = dtl.count_vector(vec, 5)
        assert count == vec.local_size
        count_zero = dtl.count_vector(vec, 0)
        assert count_zero == 0

    def test_count_if_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = float(i)
        count = dtl.count_if_vector(vec, lambda x: x > 2.0)
        expected = sum(1 for i in range(len(local)) if float(i) > 2.0)
        assert count == expected

    def test_reduce_local_vector_sum(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.float64, fill=1.0)
        result = dtl.reduce_local_vector(vec, op="sum")
        assert result == float(vec.local_size)

    def test_reduce_local_vector_min_max(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.int32)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = i * 10
        min_val = dtl.reduce_local_vector(vec, op="min")
        max_val = dtl.reduce_local_vector(vec, op="max")
        assert min_val == 0
        assert max_val == (len(local) - 1) * 10

    def test_find_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.int32, fill=0)
        idx = dtl.find_vector(vec, 999)
        assert idx is None

    def test_minmax_vector(self, context):
        vec = dtl.DistributedVector(context, size=10, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = i * 5.0 - 10.0
        min_val, max_val = dtl.minmax_vector(vec)
        assert min_val == -10.0
        assert max_val == (len(local) - 1) * 5.0 - 10.0


class TestSort:
    """Test sort operations under MPI."""

    def test_sort_vector_ascending(self, context):
        vec = dtl.DistributedVector(context, size=20, dtype=np.int32)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = len(local) - i
        dtl.sort_vector(vec)
        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i - 1] <= sorted_local[i]

    def test_sort_vector_descending(self, context):
        vec = dtl.DistributedVector(context, size=20, dtype=np.float64)
        local = vec.local_view()
        for i in range(len(local)):
            local[i] = float(i)
        dtl.sort_vector(vec, reverse=True)
        sorted_local = vec.local_view()
        for i in range(1, len(sorted_local)):
            assert sorted_local[i - 1] >= sorted_local[i]


class TestScan:
    """Test scan (prefix sum) operations under MPI."""

    def test_inclusive_scan_sum(self, context):
        vec = dtl.DistributedVector(context, size=5, dtype=np.float64, fill=1.0)
        dtl.inclusive_scan_vector(vec, op="sum")
        local = vec.local_view()
        # After inclusive scan of all-ones: [1, 2, 3, 4, 5]
        for i in range(len(local)):
            assert local[i] == float(i + 1)

    def test_exclusive_scan_sum(self, context):
        vec = dtl.DistributedVector(context, size=5, dtype=np.float64, fill=1.0)
        dtl.exclusive_scan_vector(vec, op="sum")
        local = vec.local_view()
        # After exclusive scan of all-ones: [0, 1, 2, 3, 4]
        for i in range(len(local)):
            assert local[i] == float(i)


# =============================================================================
# Distributed Vector + Collective Integration Tests
# =============================================================================


class TestVectorCollectiveIntegration:
    """Test combining DistributedVector with collectives."""

    def test_vector_local_sum_allreduce(self, context):
        """Sum local vector data, allreduce to get global sum."""
        vec = dtl.DistributedVector(context, size=100, dtype=np.float64, fill=1.0)
        local = vec.local_view()
        local_sum = np.sum(local)
        global_sum = dtl.allreduce(context, local_sum, op="sum")
        assert abs(global_sum - 100.0) < 1e-10

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_rank_data_allreduce(self, mpi_context):
        """Each rank fills with rank value, allreduce sum gives expected result."""
        size_per_rank = 10
        total_size = size_per_rank * mpi_context.size
        vec = dtl.DistributedVector(mpi_context, size=total_size, dtype=np.float64)
        local = vec.local_view()
        local[:] = float(mpi_context.rank)

        local_sum = np.array([np.sum(local)], dtype=np.float64)
        global_sum = dtl.allreduce(mpi_context, local_sum, op="sum")

        # Sum = sum(rank * size_per_rank for rank in range(n))
        n = mpi_context.size
        expected = size_per_rank * n * (n - 1) / 2.0
        assert abs(global_sum[0] - expected) < 1e-10

    @pytest.mark.multirank
    @pytest.mark.mpi
    def test_broadcast_fill_verify(self, mpi_context):
        """Root broadcasts a fill value, all ranks apply it."""
        if mpi_context.rank == 0:
            val = np.array([42.0], dtype=np.float64)
        else:
            val = np.zeros(1, dtype=np.float64)
        val = dtl.broadcast(mpi_context, val, root=0)

        vec = dtl.DistributedVector(mpi_context, size=50, dtype=np.float64)
        vec.fill(val[0])
        local = vec.local_view()
        assert np.all(local == 42.0)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error paths and exception handling."""

    def test_unsupported_dtype_raises(self, context):
        with pytest.raises(TypeError):
            dtl.DistributedVector(context, size=10, dtype=np.complex128)

    def test_dtl_error_exists(self):
        assert hasattr(dtl, "DTLError")
        assert issubclass(dtl.DTLError, Exception)

    def test_communication_error_exists(self):
        assert hasattr(dtl, "CommunicationError")
        assert issubclass(dtl.CommunicationError, dtl.DTLError)
