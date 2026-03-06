// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_multirank.cpp
/// @brief MPI multi-rank integration tests for DTL containers and algorithms
/// @details Exercises distributed_vector, distributed_tensor, distributed_map,
///          and collective/point-to-point communication under real MPI.
///          Run with: mpirun -n 2 ./test_mpi_multirank
///                    mpirun -n 4 ./test_mpi_multirank

#include <dtl/core/config.hpp>

#if DTL_ENABLE_MPI

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_tensor.hpp>
#include <dtl/containers/distributed_map.hpp>
#include <dtl/algorithms/algorithms.hpp>
#include <dtl/communication/communication.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

namespace dtl::integration_test {

// ============================================================================
// Global MPI state
// ============================================================================

static dtl::environment* g_env = nullptr;

// ============================================================================
// Test Fixture
// ============================================================================

class MpiMultiRankTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx = g_env->make_world_context();
        rank_ = ctx.rank();
        size_ = ctx.size();
        comm_ = ctx.get<dtl::mpi_domain>().communicator();
        if (size_ < 2) {
            GTEST_SKIP() << "MPI multi-rank tests require at least 2 ranks";
            return;
        }
    }

    dtl::mpi::mpi_comm_adapter comm_{};
    dtl::rank_t rank_{0};
    dtl::rank_t size_{1};
};

// ============================================================================
// distributed_vector Tests
// ============================================================================

TEST_F(MpiMultiRankTest, DistributedVectorCreation) {
    // Create a distributed vector partitioned across ranks
    constexpr dtl::size_type global_size = 1000;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(global_size, mpi);

    EXPECT_EQ(vec.global_size(), global_size);
    EXPECT_GT(vec.local_size(), 0u);

    // Sum of local sizes across all ranks should equal global size
    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(vec.local_size()));
    EXPECT_EQ(static_cast<dtl::size_type>(total), global_size);
}

TEST_F(MpiMultiRankTest, DistributedVectorFillAndAllreduce) {
    constexpr dtl::size_type global_size = 400;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(global_size, mpi);

    // Each rank fills its local partition with rank+1
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(rank_ + 1);
    }

    // Reduce (sum) across ranks — result is valid on ALL ranks (allreduce)
    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, comm_);

    // Expected: sum over each rank's contribution
    // rank r contributes local_size(r) * (r+1)
    int expected = 0;
    for (dtl::rank_t r = 0; r < size_; ++r) {
        expected += static_cast<int>(vec.local_size_for_rank(r)) * (r + 1);
    }

    EXPECT_EQ(global_sum, expected);
}

TEST_F(MpiMultiRankTest, DistributedVectorAllreduceSpan) {
    // Use the communication layer allreduce directly (span-based)
    constexpr int count = 4;
    std::vector<int> send(count, static_cast<int>(rank_ + 1));
    std::vector<int> recv(count, 0);

    dtl::allreduce(comm_,
                   std::span<const int>(send),
                   std::span<int>(recv),
                   dtl::reduce_sum<>{});

    // Each element should be sum of (r+1) for all ranks = size*(size+1)/2
    int expected_val = static_cast<int>(size_ * (size_ + 1) / 2);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(recv[i], expected_val) << "index=" << i;
    }
}

// ============================================================================
// distributed_tensor Tests
// ============================================================================

TEST_F(MpiMultiRankTest, DistributedTensor2DCreation) {
    dtl::mpi_domain mpi;
    dtl::nd_extent<2> extents{10, 20};  // 10x20 matrix
    dtl::distributed_tensor<double, 2> tensor(extents, mpi);

    EXPECT_EQ(tensor.global_size(), 200u);
    EXPECT_GT(tensor.local_size(), 0u);

    // Fill local data
    auto local = tensor.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<double>(rank_) + 0.5;
    }

    // Sum of local sizes must equal global size
    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(tensor.local_size()));
    EXPECT_EQ(static_cast<dtl::size_type>(total), 200u);
}

// ============================================================================
// distributed_map Tests
// ============================================================================

TEST_F(MpiMultiRankTest, DistributedMapLocalInsert) {
    dtl::mpi_domain mpi;
    dtl::distributed_map<int, int> dmap(mpi);

    // Each rank inserts local keys: rank*100 .. rank*100 + 9
    for (int i = 0; i < 10; ++i) {
        int key = static_cast<int>(rank_) * 100 + i;
        dmap.insert(key, key * 10);
    }

    auto flush_result = dmap.flush_pending_with_comm(comm_);
    ASSERT_TRUE(flush_result) << flush_result.error().message();

    // Allreduce total size
    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(dmap.local_size()));
    EXPECT_EQ(total, static_cast<long>(size_) * 10);
}

// ============================================================================
// Algorithm Tests
// ============================================================================

TEST_F(MpiMultiRankTest, ForEachLocal) {
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(200, 0, mpi);

    // for_each is local — doubles each element
    dtl::for_each(dtl::seq{}, vec, [](int& x) { x = 42; });

    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST_F(MpiMultiRankTest, TransformLocal) {
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> src(200, 1, mpi);
    dtl::distributed_vector<int> dst(200, 0, mpi);

    dtl::transform(dtl::seq{}, src, dst, [](int x) { return x * 3; });

    auto local = dst.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 3);
    }
}

TEST_F(MpiMultiRankTest, ReduceAcrossRanks) {
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 100;
    dtl::distributed_vector<int> vec(N, 1, mpi);

    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, comm_);

    // Every element is 1, so sum == global_size
    EXPECT_EQ(global_sum, static_cast<int>(N));
}

TEST_F(MpiMultiRankTest, SortAcrossRanks) {
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 200;
    dtl::distributed_vector<int> vec(N, mpi);

    // Fill: each rank fills descending within its partition
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        // Assign values that interleave across ranks
        local[i] = static_cast<int>(N) - static_cast<int>(rank_ * local.size() + i);
    }

    // Global sort
    auto sort_result = dtl::sort(dtl::seq{}, vec, std::less<>{}, comm_);
    EXPECT_TRUE(sort_result.success);

    // Verify locally sorted
    auto sorted_local = vec.local_view();
    for (dtl::size_type i = 1; i < sorted_local.size(); ++i) {
        EXPECT_LE(sorted_local[i - 1], sorted_local[i])
            << "rank=" << rank_ << " local index " << i;
    }

    // Verify cross-rank ordering: last element on rank r <= first element on rank r+1
    if (size_ > 1 && sorted_local.size() > 0) {
        int my_last = sorted_local[sorted_local.size() - 1];
        int my_first = sorted_local[0];

        if (rank_ < size_ - 1) {
            // Send last to rank+1
            comm_.send(&my_last, sizeof(int), rank_ + 1, 100);
        }
        if (rank_ > 0) {
            int prev_last = 0;
            comm_.recv(&prev_last, sizeof(int), rank_ - 1, 100);
            EXPECT_LE(prev_last, my_first)
                << "Cross-rank ordering violated between rank " << (rank_ - 1)
                << " and rank " << rank_;
        }
    }
}

TEST_F(MpiMultiRankTest, InclusiveScanAcrossRanks) {
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 100;
    dtl::distributed_vector<int> input(N, 1, mpi);   // all 1s
    dtl::distributed_vector<int> output(N, 0, mpi);

    auto res = dtl::inclusive_scan(dtl::seq{}, input, output, 0, std::plus<>{}, comm_);
    EXPECT_TRUE(res.has_value()) << "inclusive_scan should succeed";

    // The last element of the global scan should equal N
    auto out_local = output.local_view();
    if (rank_ == size_ - 1 && out_local.size() > 0) {
        EXPECT_EQ(out_local[out_local.size() - 1], static_cast<int>(N));
    }

    // Verify monotonically increasing locally
    for (dtl::size_type i = 1; i < out_local.size(); ++i) {
        EXPECT_GE(out_local[i], out_local[i - 1]);
    }
}

// ============================================================================
// Point-to-Point Communication Tests
// ============================================================================

TEST_F(MpiMultiRankTest, SendRecvBetweenPairs) {
    // Pair: rank 0 sends to rank 1
    if (size_ < 2) GTEST_SKIP();

    constexpr int tag = 42;
    int value = 0;

    if (rank_ == 0) {
        value = 12345;
        dtl::send(comm_, value, /*dest=*/1, tag);
    } else if (rank_ == 1) {
        dtl::recv(comm_, value, /*source=*/0, tag);
        EXPECT_EQ(value, 12345);
    }
    comm_.barrier();
}

TEST_F(MpiMultiRankTest, SendRecvSpan) {
    if (size_ < 2) GTEST_SKIP();

    constexpr int count = 10;
    constexpr int tag = 43;
    std::vector<int> data(count);

    if (rank_ == 0) {
        std::iota(data.begin(), data.end(), 100);
        dtl::send(comm_, std::span<const int>(data), /*dest=*/1, tag);
    } else if (rank_ == 1) {
        dtl::recv(comm_, std::span<int>(data), /*source=*/0, tag);
        for (int i = 0; i < count; ++i) {
            EXPECT_EQ(data[i], 100 + i);
        }
    }
    comm_.barrier();
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(MpiMultiRankTest, BroadcastSingleValue) {
    int value = (rank_ == 0) ? 999 : 0;
    dtl::broadcast(comm_, value, /*root=*/0);
    EXPECT_EQ(value, 999);
}

TEST_F(MpiMultiRankTest, BroadcastSpan) {
    constexpr int count = 8;
    std::vector<int> data(count, 0);
    if (rank_ == 0) {
        std::iota(data.begin(), data.end(), 10);
    }

    dtl::broadcast(comm_, std::span<int>(data), /*root=*/0);

    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(data[i], 10 + i);
    }
}

// ============================================================================
// Gather / Scatter Tests
// ============================================================================

TEST_F(MpiMultiRankTest, GatherToRoot) {
    // Each rank sends one int: its rank
    int send_val = static_cast<int>(rank_);
    std::vector<int> recv_buf(static_cast<size_t>(size_), 0);

    dtl::gather(comm_,
                std::span<const int>(&send_val, 1),
                std::span<int>(recv_buf),
                /*root=*/0);

    if (rank_ == 0) {
        for (dtl::rank_t r = 0; r < size_; ++r) {
            EXPECT_EQ(recv_buf[r], static_cast<int>(r));
        }
    }
}

TEST_F(MpiMultiRankTest, ScatterFromRoot) {
    // Root scatters rank indices
    std::vector<int> send_buf;
    if (rank_ == 0) {
        send_buf.resize(static_cast<size_t>(size_));
        std::iota(send_buf.begin(), send_buf.end(), 100);
    }

    int recv_val = 0;
    dtl::scatter(comm_,
                 std::span<const int>(send_buf),
                 std::span<int>(&recv_val, 1),
                 /*root=*/0);

    EXPECT_EQ(recv_val, 100 + static_cast<int>(rank_));
}

// ============================================================================
// Barrier Test
// ============================================================================

TEST_F(MpiMultiRankTest, BarrierSynchronization) {
    // Trivially test that barrier completes without deadlock
    for (int i = 0; i < 5; ++i) {
        comm_.barrier();
    }
    SUCCEED() << "Barrier completed successfully " << 5 << " times";
}

// ============================================================================
// Communicator Split Tests
// ============================================================================

TEST_F(MpiMultiRankTest, CommunicatorSplit) {
    // Split even/odd ranks
    dtl::mpi_domain mpi;
    int color = static_cast<int>(rank_ % 2);  // 0=even, 1=odd

    auto split_result = mpi.split(color);
    ASSERT_TRUE(split_result.has_value()) << "Communicator split should succeed";

    auto sub = std::move(*split_result);
    EXPECT_TRUE(sub.valid());

    // Verify sub-communicator sizes
    dtl::rank_t expected_sub_size = 0;
    for (dtl::rank_t r = 0; r < size_; ++r) {
        if (static_cast<int>(r % 2) == color) ++expected_sub_size;
    }
    EXPECT_EQ(sub.size(), expected_sub_size);

    // Barrier on sub-communicator
    sub.barrier();
}

TEST_F(MpiMultiRankTest, SubcommunicatorCollective) {
    // Split into two halves and perform allreduce within each half
    dtl::mpi_domain mpi;
    int color = (rank_ < size_ / 2) ? 0 : 1;
    auto split_result = mpi.split(color);
    ASSERT_TRUE(split_result.has_value());

    auto sub = std::move(*split_result);
    auto& sub_comm = sub.communicator();

    // Each sub-rank contributes 1; sum should equal sub-communicator size
    int local_val = 1;
    int global_sum = sub_comm.template allreduce_sum_value<int>(local_val);
    EXPECT_EQ(global_sum, static_cast<int>(sub.size()));
}

}  // namespace dtl::integration_test

// ============================================================================
// MPI-aware main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    dtl::integration_test::g_env = new dtl::environment(argc, argv);

    // Only rank 0 should print test results to avoid interleaved output
    auto ctx = dtl::integration_test::g_env->make_world_context();
    if (ctx.rank() != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    delete dtl::integration_test::g_env;
    dtl::integration_test::g_env = nullptr;

    return result;
}

#else  // !DTL_ENABLE_MPI

#include <gtest/gtest.h>

TEST(MpiMultiRankTest, SkippedMpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled — skipping multi-rank integration tests";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_MPI
