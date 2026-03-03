// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_scan_mpi.cpp
/// @brief Integration tests for distributed scan algorithms with MPI
/// @details Tests cross-rank scan operations across multiple MPI ranks.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests
///       or:       mpirun -np 4 ./dtl_mpi_tests

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/reductions/scan.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <functional>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture for Scan MPI Tests
// =============================================================================

class ScanMpiTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<mpi::mpi_comm_adapter>();
        rank_ = comm_->rank();
        size_ = comm_->size();
    }

    std::unique_ptr<mpi::mpi_comm_adapter> comm_;
    rank_t rank_ = 0;
    rank_t size_ = 1;
};

// =============================================================================
// Cross-Rank Inclusive Scan Tests
// =============================================================================

TEST_F(ScanMpiTest, InclusiveScanTwoRanks) {
    // Each rank contributes 2 elements
    distributed_vector<int> input(4, 1, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill with values: rank 0 gets [1,2], rank 1 gets [3,4]
    auto local_in = input.local_view();
    index_t offset = input.global_offset();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(offset + static_cast<index_t>(i) + 1);
    }

    auto result = inclusive_scan(seq{}, input, output, 0, std::plus<>{}, *comm_);
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    if (rank_ == 0) {
        // Rank 0: input=[1,2], output=[1,3]
        EXPECT_EQ(local_out[0], 1);   // 1
        EXPECT_EQ(local_out[1], 3);   // 1+2
    } else if (rank_ == 1) {
        // Rank 1: input=[3,4], output=[6,10]
        EXPECT_EQ(local_out[0], 6);   // 1+2+3
        EXPECT_EQ(local_out[1], 10);  // 1+2+3+4
    }
}

TEST_F(ScanMpiTest, InclusiveScanFourRanks) {
    if (size_ < 4) {
        GTEST_SKIP() << "Test requires 4 MPI ranks";
    }

    // Each rank contributes 1 element
    distributed_vector<int> input(4, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill: rank i gets value i+1
    auto local_in = input.local_view();
    if (local_in.size() > 0) {
        local_in[0] = rank_ + 1;
    }

    auto result = inclusive_scan(seq{}, input, output, 0, std::plus<>{}, *comm_);
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    if (local_out.size() > 0) {
        // Expected: rank 0->1, rank 1->3, rank 2->6, rank 3->10
        int expected = (rank_ + 1) * (rank_ + 2) / 2;
        EXPECT_EQ(local_out[0], expected);
    }
}

// =============================================================================
// Cross-Rank Exclusive Scan Tests
// =============================================================================

TEST_F(ScanMpiTest, ExclusiveScanTwoRanks) {
    // Each rank contributes 2 elements
    distributed_vector<int> input(4, 1, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill with values: rank 0 gets [1,2], rank 1 gets [3,4]
    auto local_in = input.local_view();
    index_t offset = input.global_offset();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(offset + static_cast<index_t>(i) + 1);
    }

    auto result = exclusive_scan(seq{}, input, output, 0, std::plus<>{}, *comm_);
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    if (rank_ == 0) {
        // Rank 0: input=[1,2], output=[0,1]
        EXPECT_EQ(local_out[0], 0);   // init
        EXPECT_EQ(local_out[1], 1);   // 0+1
    } else if (rank_ == 1) {
        // Rank 1: input=[3,4], output=[3,6]
        EXPECT_EQ(local_out[0], 3);   // 0+1+2
        EXPECT_EQ(local_out[1], 6);   // 0+1+2+3
    }
}

// =============================================================================
// Cross-Rank Transform Inclusive Scan Tests
// =============================================================================

TEST_F(ScanMpiTest, TransformInclusiveScanTwoRanks) {
    // Each rank contributes 2 elements
    distributed_vector<int> input(4, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill with values: rank 0 gets [1,2], rank 1 gets [3,4]
    auto local_in = input.local_view();
    index_t offset = input.global_offset();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(offset + static_cast<index_t>(i) + 1);
    }

    // Transform: square each value
    auto result = transform_inclusive_scan(
        seq{}, input, output, 0,
        std::plus<>{},
        [](int x) { return x * x; },
        *comm_
    );
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    if (rank_ == 0) {
        // Rank 0: input=[1,2], transformed=[1,4], output=[1,5]
        EXPECT_EQ(local_out[0], 1);   // 1^2
        EXPECT_EQ(local_out[1], 5);   // 1+4
    } else if (rank_ == 1) {
        // Rank 1: input=[3,4], transformed=[9,16], output=[14,30]
        EXPECT_EQ(local_out[0], 14);  // 1+4+9
        EXPECT_EQ(local_out[1], 30);  // 1+4+9+16
    }
}

TEST_F(ScanMpiTest, TransformInclusiveScanMultiplication) {
    // Each rank contributes 2 elements
    distributed_vector<int> input(4, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill with 1s
    auto local_in = input.local_view();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = 1;
    }

    // Transform: double each value, then sum
    auto result = transform_inclusive_scan(
        seq{}, input, output, 0,
        std::plus<>{},
        [](int x) { return x * 2; },
        *comm_
    );
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    // Each element contributes 2, so cumulative should be 2*(index+1)
    index_t offset = input.global_offset();
    for (size_type i = 0; i < local_out.size(); ++i) {
        index_t expected_index = offset + static_cast<index_t>(i) + 1;
        int expected = static_cast<int>(expected_index * 2);
        EXPECT_EQ(local_out[i], expected);
    }
}

// =============================================================================
// Cross-Rank Transform Exclusive Scan Tests
// =============================================================================

TEST_F(ScanMpiTest, TransformExclusiveScanTwoRanks) {
    // Each rank contributes 2 elements
    distributed_vector<int> input(4, *comm_);
    distributed_vector<int> output(4, *comm_);

    // Fill with values: rank 0 gets [1,2], rank 1 gets [3,4]
    auto local_in = input.local_view();
    index_t offset = input.global_offset();
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_in[i] = static_cast<int>(offset + static_cast<index_t>(i) + 1);
    }

    // Transform: square each value
    auto result = transform_exclusive_scan(
        seq{}, input, output, 0,
        std::plus<>{},
        [](int x) { return x * x; },
        *comm_
    );
    EXPECT_TRUE(result.has_value());

    auto local_out = output.local_view();

    if (rank_ == 0) {
        // Rank 0: input=[1,2], transformed=[1,4], output=[0,1]
        EXPECT_EQ(local_out[0], 0);   // init
        EXPECT_EQ(local_out[1], 1);   // 0+1
    } else if (rank_ == 1) {
        // Rank 1: input=[3,4], transformed=[9,16], output=[5,14]
        EXPECT_EQ(local_out[0], 5);   // 0+1+4
        EXPECT_EQ(local_out[1], 14);  // 0+1+4+9
    }
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
