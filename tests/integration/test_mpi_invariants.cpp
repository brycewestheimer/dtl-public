// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_invariants.cpp
/// @brief Algebraic invariant tests for MPI correctness
/// @details Tests mathematical invariants that must hold regardless of rank count.
///          Designed to run at 2, 3, and 4 ranks to cover non-power-of-2 edge cases.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_MPI

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/algorithms.hpp>
#include <dtl/communication/communication.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace dtl::invariant_test {

static dtl::environment* g_env = nullptr;

class MpiInvariantTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto ctx = g_env->make_world_context();
        rank_ = ctx.rank();
        size_ = ctx.size();
        comm_ = ctx.get<dtl::mpi_domain>().communicator();
        if (size_ < 2) {
            GTEST_SKIP() << "Invariant tests require at least 2 ranks";
        }
    }

    dtl::mpi::mpi_comm_adapter comm_{};
    dtl::rank_t rank_{0};
    dtl::rank_t size_{1};
};

// ============================================================================
// Partition Coverage: sum(local_size) == global_size
// ============================================================================

TEST_F(MpiInvariantTest, PartitionCoverageVector) {
    constexpr dtl::size_type global_size = 1000;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(global_size, mpi);

    EXPECT_GT(vec.local_size(), 0u);

    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(vec.local_size()));
    EXPECT_EQ(static_cast<dtl::size_type>(total), global_size)
        << "Partition coverage violated: sum(local_size) != global_size"
        << " with " << size_ << " ranks";
}

TEST_F(MpiInvariantTest, PartitionCoverageOddSize) {
    // Odd global size — tests remainder distribution across ranks
    constexpr dtl::size_type global_size = 1001;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(global_size, mpi);

    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(vec.local_size()));
    EXPECT_EQ(static_cast<dtl::size_type>(total), global_size)
        << "Partition coverage violated for odd global_size=" << global_size
        << " with " << size_ << " ranks";
}

TEST_F(MpiInvariantTest, PartitionCoveragePrime) {
    // Prime global size — no rank count divides evenly
    constexpr dtl::size_type global_size = 997;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(global_size, mpi);

    auto total = comm_.template allreduce_sum_value<long>(
        static_cast<long>(vec.local_size()));
    EXPECT_EQ(static_cast<dtl::size_type>(total), global_size)
        << "Partition coverage violated for prime global_size=" << global_size
        << " with " << size_ << " ranks";
}

// ============================================================================
// Allreduce Correctness: allreduce(rank+1, sum) == P*(P+1)/2
// ============================================================================

TEST_F(MpiInvariantTest, AllreduceSumFormula) {
    int local_val = static_cast<int>(rank_ + 1);
    int result = comm_.template allreduce_sum_value<int>(local_val);

    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;
    EXPECT_EQ(result, expected)
        << "Allreduce invariant violated: sum(rank+1) != P*(P+1)/2"
        << " with P=" << size_;
}

TEST_F(MpiInvariantTest, AllreduceSpanSum) {
    constexpr int count = 16;
    std::vector<int> send(count, static_cast<int>(rank_ + 1));
    std::vector<int> recv(count, 0);

    dtl::allreduce(comm_,
                   std::span<const int>(send),
                   std::span<int>(recv),
                   dtl::reduce_sum<>{});

    int expected = static_cast<int>(size_) * (static_cast<int>(size_) + 1) / 2;
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(recv[i], expected) << "index=" << i;
    }
}

// ============================================================================
// Sort Ordering: each rank's max <= next rank's min
// ============================================================================

TEST_F(MpiInvariantTest, SortCrossRankOrdering) {
    constexpr dtl::size_type N = 300;
    dtl::mpi_domain mpi;
    dtl::distributed_vector<int> vec(N, mpi);

    // Fill with values that interleave across ranks
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(N) - static_cast<int>(rank_ * local.size() + i);
    }

    auto sort_result = dtl::sort(dtl::seq{}, vec, std::less<>{}, comm_);
    EXPECT_TRUE(sort_result.success);

    // Verify locally sorted
    auto sorted = vec.local_view();
    for (dtl::size_type i = 1; i < sorted.size(); ++i) {
        EXPECT_LE(sorted[i - 1], sorted[i])
            << "rank=" << rank_ << " local index " << i;
    }

    // Verify cross-rank ordering
    if (sorted.size() > 0) {
        int my_last = sorted[sorted.size() - 1];
        int my_first = sorted[0];

        if (rank_ < size_ - 1) {
            comm_.send(&my_last, sizeof(int), rank_ + 1, 200);
        }
        if (rank_ > 0) {
            int prev_last = 0;
            comm_.recv(&prev_last, sizeof(int), rank_ - 1, 200);
            EXPECT_LE(prev_last, my_first)
                << "Cross-rank ordering violated between rank "
                << (rank_ - 1) << " and rank " << rank_
                << " (prev_last=" << prev_last << ", my_first=" << my_first << ")"
                << " with " << size_ << " total ranks";
        }
    }
}

// ============================================================================
// Broadcast Identity: all ranks identical after broadcast
// ============================================================================

TEST_F(MpiInvariantTest, BroadcastIdentityScalar) {
    int value = (rank_ == 0) ? 42 : -1;
    dtl::broadcast(comm_, value, /*root=*/0);
    EXPECT_EQ(value, 42)
        << "Broadcast identity violated on rank " << rank_;
}

TEST_F(MpiInvariantTest, BroadcastIdentitySpan) {
    constexpr int count = 64;
    std::vector<int> data(count, 0);
    if (rank_ == 0) {
        std::iota(data.begin(), data.end(), 100);
    }

    dtl::broadcast(comm_, std::span<int>(data), /*root=*/0);

    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(data[i], 100 + i)
            << "Broadcast identity violated at index " << i
            << " on rank " << rank_;
    }
}

TEST_F(MpiInvariantTest, BroadcastFromNonZeroRoot) {
    // Broadcast from last rank — exercises non-root=0 path
    dtl::rank_t root = size_ - 1;
    int value = (rank_ == root) ? 777 : -1;
    dtl::broadcast(comm_, value, root);
    EXPECT_EQ(value, 777)
        << "Broadcast from rank " << root << " failed on rank " << rank_;
}

}  // namespace dtl::invariant_test

// ============================================================================
// MPI-aware main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    dtl::invariant_test::g_env = new dtl::environment(argc, argv);

    auto ctx = dtl::invariant_test::g_env->make_world_context();
    if (ctx.rank() != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    delete dtl::invariant_test::g_env;
    dtl::invariant_test::g_env = nullptr;

    return result;
}

#else  // !DTL_ENABLE_MPI

#include <gtest/gtest.h>

TEST(MpiInvariantTest, SkippedMpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled — skipping invariant integration tests";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_MPI
