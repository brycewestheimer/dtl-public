// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_sort.cpp
/// @brief Integration tests for distributed sorting algorithms with MPI
/// @details Tests sample sort, stable_sort, nth_element, partial_sort, and unique
///          with MPI communicator across multiple ranks.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests --gtest_filter="*MpiSort*"
///       or:       mpirun -np 4 ./dtl_mpi_tests --gtest_filter="*MpiSort*"

#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/algorithms/sorting/nth_element.hpp>
#include <dtl/algorithms/sorting/partial_sort.hpp>
#include <dtl/algorithms/sorting/unique.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture
// =============================================================================

class MpiSortTest : public ::testing::Test {
protected:
    void SetUp() override {
        mpi_domain_ = std::make_unique<mpi_domain>();
        adapter_ = &mpi_domain_->communicator();
    }

    std::unique_ptr<mpi_domain> mpi_domain_;
    mpi::mpi_comm_adapter* adapter_ = nullptr;
};

// =============================================================================
// Sample Sort Tests
// =============================================================================

TEST_F(MpiSortTest, SampleSortBasic) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    // Create a distributed vector with unsorted data
    // Each rank gets 10 elements in reverse order relative to rank
    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with reverse-sorted data: rank 0 gets highest values, last rank gets lowest
    auto local_v = vec.local_view();
    int base_value = (num_ranks - my_rank - 1) * static_cast<int>(elements_per_rank);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(elements_per_rank - i - 1);
    }

    // Sort with communicator
    auto result = sort(par{}, vec, std::less<>{}, *adapter_);

    EXPECT_TRUE(result.success);

    // Verify local partition is sorted
    EXPECT_TRUE(std::is_sorted(local_v.begin(), local_v.end()));

    // Verify global sort order using is_sorted with communicator
    EXPECT_TRUE(is_sorted(seq{}, vec, std::less<>{}, *adapter_));
}

TEST_F(MpiSortTest, SampleSortWithDuplicates) {
    rank_t num_ranks = adapter_->size();

    size_type elements_per_rank = 20;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with data that has many duplicates
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        // Values 0-9 with duplicates
        local_v[i] = static_cast<int>(i % 10);
    }

    // Sort
    auto result = sort(par{}, vec, std::less<>{}, *adapter_);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(std::is_sorted(local_v.begin(), local_v.end()));
}

TEST_F(MpiSortTest, SampleSortEmpty) {
    // Empty vector
    distributed_vector<int> vec(0, *adapter_);

    auto result = sort(par{}, vec, std::less<>{}, *adapter_);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.elements_sent, 0u);
    EXPECT_EQ(result.elements_received, 0u);
}

TEST_F(MpiSortTest, SampleSortDescending) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with ascending data
    auto local_v = vec.local_view();
    int base_value = my_rank * static_cast<int>(elements_per_rank);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(i);
    }

    // Sort in descending order
    auto result = sort(par{}, vec, std::greater<>{}, *adapter_);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(std::is_sorted(local_v.begin(), local_v.end(), std::greater<>{}));
}

// =============================================================================
// Stable Sort Tests
// =============================================================================

TEST_F(MpiSortTest, StableSortBasic) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with random data
    auto local_v = vec.local_view();
    std::mt19937 gen(42u + static_cast<std::mt19937::result_type>(my_rank));
    std::uniform_int_distribution<int> dist(0, 100);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = dist(gen);
    }

    // Stable sort
    auto result = stable_sort(par{}, vec, std::less<>{}, *adapter_);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(std::is_sorted(local_v.begin(), local_v.end()));
}

// =============================================================================
// is_sorted Tests
// =============================================================================

TEST_F(MpiSortTest, IsSortedTrueWhenSorted) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with globally sorted data
    auto local_v = vec.local_view();
    int base_value = my_rank * static_cast<int>(elements_per_rank);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(i);
    }

    EXPECT_TRUE(is_sorted(seq{}, vec, std::less<>{}, *adapter_));
}

TEST_F(MpiSortTest, IsSortedFalseWhenLocalUnsorted) {
    rank_t num_ranks = adapter_->size();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with locally unsorted data
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = static_cast<int>(local_v.size() - i);
    }

    EXPECT_FALSE(is_sorted(seq{}, vec, std::less<>{}, *adapter_));
}

TEST_F(MpiSortTest, IsSortedFalseWhenBoundaryViolation) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill so each rank is locally sorted but boundaries are violated
    // Rank 0: [90-99], Rank 1: [0-9], etc.
    auto local_v = vec.local_view();
    int base_value = (num_ranks - my_rank - 1) * static_cast<int>(elements_per_rank) * 10;
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(i);
    }

    EXPECT_FALSE(is_sorted(seq{}, vec, std::less<>{}, *adapter_));
}

// =============================================================================
// nth_element Tests
// =============================================================================

TEST_F(MpiSortTest, NthElementBasic) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with values 0 to global_size-1, shuffled
    auto local_v = vec.local_view();
    int base_value = my_rank * static_cast<int>(elements_per_rank);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(i);
    }

    // Shuffle local data
    std::mt19937 gen(42u + static_cast<std::mt19937::result_type>(my_rank));
    std::shuffle(local_v.begin(), local_v.end(), gen);

    // Find the median (middle element)
    index_t n = static_cast<index_t>(global_size / 2);
    auto result = nth_element(par{}, vec, n, std::less<>{}, *adapter_);

    EXPECT_TRUE(result.valid);
    // The nth element should be n (since we had values 0 to global_size-1)
    EXPECT_EQ(result.value, n);
}

// =============================================================================
// unique Tests
// =============================================================================

TEST_F(MpiSortTest, UniqueRemovesDuplicates) {
    rank_t num_ranks = adapter_->size();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with sorted data that has duplicates
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = static_cast<int>(i / 2);  // Creates pairs: 0,0,1,1,2,2,...
    }

    auto result = unique(par{}, vec, std::equal_to<>{}, *adapter_);

    EXPECT_TRUE(result.value().success);
    EXPECT_GT(result.value().removed_count, 0u);
}

TEST_F(MpiSortTest, UniqueHandlesBoundaryDuplicates) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 5;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill so last element of rank i equals first element of rank i+1
    // All elements on each rank are the rank number
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = static_cast<int>(my_rank);
    }

    auto result = unique(par{}, vec, std::equal_to<>{}, *adapter_);

    EXPECT_TRUE(result.value().success);
    // After unique, we should have roughly num_ranks unique values
    // (with some boundary duplicates removed)
    EXPECT_GT(result.value().removed_count, 0u);
}

// =============================================================================
// count_duplicates Tests
// =============================================================================

TEST_F(MpiSortTest, CountDuplicatesBasic) {
    rank_t num_ranks = adapter_->size();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with sorted data that has duplicates
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = static_cast<int>(i / 2);  // Creates pairs
    }

    size_type dups = count_duplicates(seq{}, vec, std::equal_to<>{}, *adapter_);

    // Each rank has 5 duplicates (pairs), so locally we have 5 dups
    // Plus potential boundary duplicates
    EXPECT_GT(dups, 0u);
}

// =============================================================================
// has_duplicates Tests
// =============================================================================

TEST_F(MpiSortTest, HasDuplicatesTrueWhenPresent) {
    rank_t num_ranks = adapter_->size();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with some duplicates
    auto local_v = vec.local_view();
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = static_cast<int>(i / 2);
    }

    EXPECT_TRUE(has_duplicates(seq{}, vec, std::equal_to<>{}, *adapter_));
}

TEST_F(MpiSortTest, HasDuplicatesFalseWhenNone) {
    rank_t num_ranks = adapter_->size();
    rank_t my_rank = adapter_->rank();

    size_type elements_per_rank = 10;
    size_type global_size = elements_per_rank * static_cast<size_type>(num_ranks);

    distributed_vector<int> vec(global_size, *adapter_);

    // Fill with unique sorted values
    auto local_v = vec.local_view();
    int base_value = my_rank * static_cast<int>(elements_per_rank);
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base_value + static_cast<int>(i);
    }

    EXPECT_FALSE(has_duplicates(seq{}, vec, std::equal_to<>{}, *adapter_));
}

#else  // !DTL_ENABLE_MPI

// Placeholder tests when MPI is not enabled
TEST(MpiSortTest, MpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled";
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
