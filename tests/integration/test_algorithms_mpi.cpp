// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_algorithms_mpi.cpp
/// @brief Integration tests for DTL algorithms with MPI
/// @details Tests distributed algorithms across multiple MPI ranks.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests
///       or:       mpirun -np 4 ./dtl_mpi_tests

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/algorithms.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture for Algorithm Integration Tests
// =============================================================================

class AlgorithmMpiTest : public ::testing::Test {
protected:
    void SetUp() override {
        comm_ = std::make_unique<mpi::mpi_comm_adapter>();
        rank_ = comm_->rank();
        size_ = comm_->size();
    }

    /// @brief Create a distributed vector with global size
    /// @param global_size Total elements across all ranks
    /// @param value Initial value for all elements (optional)
    template <typename T>
    distributed_vector<T> make_vector(size_type global_size, T value = T{}) {
        return distributed_vector<T>(global_size, value, *comm_);
    }

    /// @brief Fill local partition with rank-specific data pattern
    /// @param vec The vector to fill
    /// @details Fills with: global_offset + local_index
    template <typename T>
    void fill_with_global_indices(distributed_vector<T>& vec) {
        auto local = vec.local_view();
        index_t offset = vec.global_offset();
        for (size_type i = 0; i < local.size(); ++i) {
            local[i] = static_cast<T>(offset + static_cast<index_t>(i));
        }
    }

    /// @brief Verify partition sizes sum to global size
    template <typename T>
    void verify_partition_sizes(const distributed_vector<T>& vec) {
        // Gather all local sizes
        std::vector<int> local_sizes(static_cast<std::size_t>(size_));
        int my_local_size = static_cast<int>(vec.local_size());

        comm_->allgather(&my_local_size, local_sizes.data(), sizeof(int));

        int total_size = std::accumulate(local_sizes.begin(), local_sizes.end(), 0);
        EXPECT_EQ(static_cast<size_type>(total_size), vec.global_size())
            << "Sum of local sizes should equal global size";
    }

    std::unique_ptr<mpi::mpi_comm_adapter> comm_;
    rank_t rank_ = 0;
    rank_t size_ = 1;
};

// =============================================================================
// Local Algorithm Tests (no communication required)
// =============================================================================

TEST_F(AlgorithmMpiTest, ForEachMultiRank) {
    // Create vector: 100 elements distributed across ranks
    auto vec = make_vector<int>(100, 1);

    // Each rank doubles its local elements
    dtl::for_each(dtl::par{}, vec, [](int& x) { x *= 2; });

    // Verify all local elements are doubled
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 2) << "Element at local index " << i << " should be 2";
    }

    // Barrier to ensure all ranks complete
    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ForEachRankSpecificMultiplication) {
    // Create vector with 100 elements
    auto vec = make_vector<int>(100, 10);

    // Each rank multiplies by (rank + 1)
    int multiplier = rank_ + 1;
    dtl::for_each(vec, [multiplier](int& x) { x *= multiplier; });

    // Verify local elements
    auto local = vec.local_view();
    int expected = 10 * multiplier;
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], expected)
            << "Rank " << rank_ << " element " << i << " should be " << expected;
    }

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, TransformUnaryMultiRank) {
    // Create source and destination vectors
    auto src = make_vector<int>(80, 5);
    auto dst = make_vector<int>(80, 0);

    // Transform: square each element
    dtl::transform(dtl::par{}, src, dst, [](int x) { return x * x; });

    // Verify all local elements are squared
    auto local = dst.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 25) << "Element at local index " << i << " should be 25";
    }

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, TransformBinaryMultiRank) {
    // Create two source vectors and destination
    auto a = make_vector<int>(60);
    auto b = make_vector<int>(60);
    auto result = make_vector<int>(60);

    // Fill with rank-dependent values
    fill_with_global_indices(a);
    fill_with_global_indices(b);

    // Transform: add vectors
    dtl::transform(a, b, result, std::plus<int>{});

    // Verify: result[i] = a[i] + b[i] = 2 * (global_offset + local_i)
    auto local_a = a.local_view();
    auto local_result = result.local_view();
    for (size_type i = 0; i < local_result.size(); ++i) {
        EXPECT_EQ(local_result[i], 2 * local_a[i]);
    }

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, SegmentedViewIsLocalCorrect) {
    auto vec = make_vector<int>(100);
    fill_with_global_indices(vec);

    // Use segmented view - only local segments should have is_local() true
    int local_count = 0;
    int remote_count = 0;
    for (auto segment : vec.segmented_view()) {
        if (segment.is_local()) {
            local_count++;
            // Verify we can iterate local segment
            for (auto& elem : segment) {
                (void)elem;  // Just iterate
            }
        } else {
            remote_count++;
        }
    }

    // In standalone distributed_vector, all segments are local
    // (communication would be needed to access remote segments)
    EXPECT_GE(local_count, 1) << "Should have at least one local segment";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, PartitionConsistency) {
    // Test that sum of local sizes equals global size
    auto vec = make_vector<int>(1000);
    verify_partition_sizes(vec);
}

TEST_F(AlgorithmMpiTest, LocalViewDoesNotCommunicate) {
    auto vec = make_vector<int>(100);
    fill_with_global_indices(vec);

    // Get local view and modify it
    auto local = vec.local_view();
    for (auto& elem : local) {
        elem += 1000;
    }

    // No barrier - if this deadlocked, local view would be communicating
    // (which it shouldn't)

    // Verify modification
    index_t offset = vec.global_offset();
    for (size_type i = 0; i < local.size(); ++i) {
        const index_t expected = offset + static_cast<index_t>(i) + 1000;
        EXPECT_EQ(local[i], static_cast<int>(expected));
    }

    // Now barrier for test completion
    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, SegmentedForEachMultiRank) {
    auto vec = make_vector<int>(100, 1);

    // Use segmented_for_each (preferred distributed pattern)
    auto result = dtl::segmented_for_each(dtl::par{}, vec, [](int& x) { x += 5; });

    EXPECT_TRUE(result.has_value()) << "segmented_for_each should succeed";

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 6) << "Element should be 1 + 5 = 6";
    }

    comm_->barrier();
}

// =============================================================================
// Reduce Algorithm Tests (with MPI communication)
// =============================================================================

TEST_F(AlgorithmMpiTest, ReduceSumAllRanks) {
    // Create vector: each rank has elements valued [rank*10, rank*10+1, ...]
    auto vec = make_vector<int>(static_cast<size_type>(size_) * 3);  // 3 elements per rank

    // Fill with rank-specific values
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = rank_ * 10 + static_cast<int>(i);
    }

    // Compute expected global sum
    // For 2 ranks: rank0: [0,1,2] sum=3, rank1: [10,11,12] sum=33, total=36
    // For 4 ranks: rank0: [0,1,2], rank1: [10,11,12], rank2: [20,21,22], rank3: [30,31,32]
    //              sums: 3 + 33 + 63 + 93 = 192
    int expected_sum = 0;
    for (rank_t r = 0; r < size_; ++r) {
        for (int i = 0; i < 3; ++i) {
            expected_sum += r * 10 + i;
        }
    }

    // Call reduce with communicator
    int global_sum = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{}, *comm_);

    // All ranks should have the same result
    EXPECT_EQ(global_sum, expected_sum)
        << "Rank " << rank_ << " global sum mismatch";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ReduceSumLargeData) {
    // Test with larger data set
    const size_type elements_per_rank = 10000;
    auto vec = make_vector<long>(static_cast<size_type>(size_) * elements_per_rank);

    // Fill with 1s
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = 1L;
    }

    // Global sum should be total number of elements
    long global_sum = dtl::reduce(dtl::par{}, vec, 0L, std::plus<>{}, *comm_);
    long expected = static_cast<long>(static_cast<size_type>(size_) * elements_per_rank);

    EXPECT_EQ(global_sum, expected);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ReduceResultType) {
    auto vec = make_vector<int>(100, 1);

    // Use distributed_reduce to get both local and global results
    auto result = dtl::distributed_reduce(dtl::par{}, vec, 0, std::plus<>{}, *comm_);

    // Global should be total count
    EXPECT_EQ(result.global_value, 100);

    // Local should be this rank's count
    EXPECT_EQ(result.local_value, static_cast<int>(vec.local_size()));

    // has_global should be true
    EXPECT_TRUE(result.has_global);

    // value() should return global
    EXPECT_EQ(result.value(), 100);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ReduceToRoot) {
    auto vec = make_vector<int>(80, 1);

    // Reduce to root rank 0
    auto result = dtl::reduce_to(dtl::par{}, vec, 0, *comm_, 0);

    // Local value should match local size
    EXPECT_EQ(result.local_value, static_cast<int>(vec.local_size()));

    // Only root should have valid global
    if (rank_ == 0) {
        EXPECT_TRUE(result.has_global);
        EXPECT_EQ(result.global_value, 80);
    } else {
        EXPECT_FALSE(result.has_global);
    }

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, LocalReduceNoMpi) {
    auto vec = make_vector<int>(100, 2);

    // local_reduce should NOT communicate
    int local_sum = dtl::local_reduce(vec, 0, std::plus<>{});

    // Should be local_size * 2
    EXPECT_EQ(local_sum, static_cast<int>(vec.local_size() * 2));

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ReduceWithDouble) {
    auto vec = make_vector<double>(100);

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = 0.5;  // Each element is 0.5
    }

    double global_sum = dtl::reduce(dtl::par{}, vec, 0.0, std::plus<>{}, *comm_);

    // Total should be 100 * 0.5 = 50.0
    EXPECT_DOUBLE_EQ(global_sum, 50.0);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, ReduceSequentialExecution) {
    auto vec = make_vector<int>(50, 3);

    // Test with sequential execution policy
    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, *comm_);

    EXPECT_EQ(global_sum, 150);  // 50 elements * 3

    comm_->barrier();
}

// =============================================================================
// Edge Case Tests
// =============================================================================

TEST_F(AlgorithmMpiTest, EmptyContainer) {
    auto vec = make_vector<int>(0);

    EXPECT_EQ(vec.global_size(), 0u);
    EXPECT_EQ(vec.local_size(), 0u);

    // Reduce on empty should return init
    int result = dtl::reduce(dtl::seq{}, vec, 42, std::plus<>{}, *comm_);
    EXPECT_EQ(result, 42);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, SingleElement) {
    // Single element distributed across ranks - only one rank has it
    auto vec = make_vector<int>(1, 7);

    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, *comm_);
    EXPECT_EQ(global_sum, 7);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, UnevenDistribution) {
    // 7 elements with 2 ranks = 4 + 3 distribution
    // 7 elements with 4 ranks = 2 + 2 + 2 + 1 distribution
    auto vec = make_vector<int>(7, 1);

    verify_partition_sizes(vec);

    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, *comm_);
    EXPECT_EQ(global_sum, 7);

    comm_->barrier();
}

// =============================================================================
// Accumulate Tests (Phase 6 Task 6.1)
// =============================================================================

TEST_F(AlgorithmMpiTest, AccumulateIntSum) {
    // Test accumulate with addition (commutative, but testing accumulate path)
    auto vec = make_vector<int>(12);  // 12 elements total
    fill_with_global_indices(vec);

    // Accumulate with communicator
    int result = dtl::accumulate(dtl::seq{}, vec, 0, std::plus<>{}, *comm_);

    // Expected: sum of 0..11 = 66
    EXPECT_EQ(result, 66) << "Rank " << rank_ << " accumulate mismatch";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, AccumulateStringConcat) {
    // Test accumulate with string concatenation (non-commutative)
    auto vec = make_vector<std::string>(static_cast<size_type>(size_) * 2);  // 2 strings per rank

    // Fill with rank-specific strings
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = std::to_string(rank_ * 100 + static_cast<int>(i));
    }

    // Accumulate with string concatenation
    auto concat_op = [](const std::string& a, const std::string& b) {
        return a.empty() ? b : a + "," + b;
    };
    std::string result = dtl::accumulate(dtl::seq{}, vec, std::string{}, concat_op, *comm_);

    // Verify result contains elements in rank order
    // For 2 ranks: "0,1,100,101"
    // For 4 ranks: "0,1,100,101,200,201,300,301"
    std::string expected;
    for (rank_t r = 0; r < size_; ++r) {
        for (int i = 0; i < 2; ++i) {
            if (!expected.empty()) expected += ",";
            expected += std::to_string(r * 100 + i);
        }
    }

    EXPECT_EQ(result, expected) << "Rank " << rank_ << " string accumulate mismatch";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, AccumulateEmptyContainer) {
    auto vec = make_vector<int>(0);
    int result = dtl::accumulate(dtl::seq{}, vec, 42, std::plus<>{}, *comm_);
    EXPECT_EQ(result, 42) << "Empty accumulate should return init";
    comm_->barrier();
}

// =============================================================================
// MinMax Element Tests (Phase 6 Task 6.1)
// =============================================================================

TEST_F(AlgorithmMpiTest, MinElementBasic) {
    auto vec = make_vector<int>(20);  // 20 elements distributed

    // Fill with rank-specific values
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = rank_ * 100 + static_cast<int>(i);
    }

    // Find global minimum
    auto result = dtl::min_element(dtl::par{}, vec, std::less<>{}, *comm_);

    // Global min should be 0 (rank 0, index 0)
    ASSERT_TRUE(result.valid) << "Result should be valid";
    EXPECT_EQ(result.value, 0) << "Global min value should be 0";
    EXPECT_EQ(result.global_index, 0) << "Global min index should be 0";
    EXPECT_EQ(result.owner_rank, 0) << "Global min should be owned by rank 0";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, MaxElementBasic) {
    auto vec = make_vector<int>(20);

    // Fill with rank-specific values
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = rank_ * 100 + static_cast<int>(i);
    }

    // Find global maximum
    auto result = dtl::max_element(dtl::par{}, vec, std::less<>{}, *comm_);

    // Global max depends on number of ranks
    // For 2 ranks with 10 elements each: max = 109 (rank 1, index 19)
    // For 4 ranks with 5 elements each: max = 304 (rank 3, index 19)
    int expected_max = (size_ - 1) * 100 + (20 / size_ - 1);

    ASSERT_TRUE(result.valid) << "Result should be valid";
    EXPECT_EQ(result.value, expected_max) << "Global max value mismatch";
    EXPECT_EQ(result.owner_rank, size_ - 1) << "Global max should be owned by last rank";

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, MinMaxElementCombined) {
    auto vec = make_vector<int>(16);
    fill_with_global_indices(vec);

    // Find both min and max
    auto result = dtl::minmax_element(dtl::par{}, vec, std::less<>{}, *comm_);

    ASSERT_TRUE(result.min.valid && result.max.valid);
    EXPECT_EQ(result.min.value, 0) << "Global min should be 0";
    EXPECT_EQ(result.max.value, 15) << "Global max should be 15";
    EXPECT_EQ(result.min.global_index, 0);
    EXPECT_EQ(result.max.global_index, 15);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, MinElementWithNegatives) {
    auto vec = make_vector<int>(20);

    // Fill with negative values decreasing by rank
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = -(rank_ * 100 + static_cast<int>(i));
    }

    auto result = dtl::min_element(dtl::par{}, vec, std::less<>{}, *comm_);

    // Most negative value will be on last rank
    int expected_min = -((size_ - 1) * 100 + (20 / size_ - 1));

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, expected_min);
    EXPECT_EQ(result.owner_rank, size_ - 1);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, MinMaxElementAllSameValue) {
    // All elements have the same value
    auto vec = make_vector<int>(20, 42);

    auto result = dtl::minmax_element(dtl::par{}, vec, std::less<>{}, *comm_);

    ASSERT_TRUE(result.min.valid && result.max.valid);
    EXPECT_EQ(result.min.value, 42);
    EXPECT_EQ(result.max.value, 42);
    // When all equal, first rank should own both
    EXPECT_EQ(result.min.owner_rank, 0);
    EXPECT_EQ(result.max.owner_rank, 0);

    comm_->barrier();
}

TEST_F(AlgorithmMpiTest, MinElementDoubleValues) {
    auto vec = make_vector<double>(24);

    // Fill with floating point values
    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<double>(rank_) + static_cast<double>(i) / 10.0;
    }

    auto result = dtl::min_element(dtl::par{}, vec, std::less<>{}, *comm_);

    ASSERT_TRUE(result.valid);
    EXPECT_DOUBLE_EQ(result.value, 0.0);
    EXPECT_EQ(result.owner_rank, 0);

    comm_->barrier();
}

#else  // !DTL_ENABLE_MPI

TEST(AlgorithmMpiTest, MpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled - skipping MPI algorithm tests";
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
