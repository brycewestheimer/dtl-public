// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_partition_algorithm.cpp
/// @brief Unit tests for distributed partition algorithm (R6.5)

#include <dtl/algorithms/modifying/partition_algorithm.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

namespace {
struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};
}  // namespace

// =============================================================================
// Basic Partition Tests
// =============================================================================

TEST(PartitionAlgorithmTest, EvenOddPartition) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(10, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < 10; ++i) {
        local[i] = static_cast<int>(i + 1);  // [1, 2, 3, ..., 10]
    }

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x % 2 == 0; });

    EXPECT_EQ(res.local_true_count, 5u);  // 5 even numbers
    EXPECT_FALSE(res.has_global);

    // Verify: first 5 elements should be even, last 5 should be odd
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(local[i] % 2, 0) << "Element at index " << i << " should be even";
    }
    for (size_type i = 5; i < 10; ++i) {
        EXPECT_NE(local[i] % 2, 0) << "Element at index " << i << " should be odd";
    }
}

TEST(PartitionAlgorithmTest, AllSatisfy) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 2, ctx);  // All even

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x % 2 == 0; });

    EXPECT_EQ(res.local_true_count, 5u);
}

TEST(PartitionAlgorithmTest, NoneSatisfy) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 1, ctx);  // All odd

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x % 2 == 0; });

    EXPECT_EQ(res.local_true_count, 0u);
}

TEST(PartitionAlgorithmTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x > 0; });

    EXPECT_EQ(res.local_true_count, 0u);
}

TEST(PartitionAlgorithmTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(1, 42, ctx);

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x > 0; });

    EXPECT_EQ(res.local_true_count, 1u);
}

TEST(PartitionAlgorithmTest, SingleElementFails) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(1, -5, ctx);

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x > 0; });

    EXPECT_EQ(res.local_true_count, 0u);
}

TEST(PartitionAlgorithmTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(6, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = -2; local[2] = 3;
    local[3] = -4; local[4] = 5; local[5] = -6;

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x > 0; });

    EXPECT_EQ(res.local_true_count, 3u);
}

// =============================================================================
// Stable Partition Tests
// =============================================================================

TEST(StablePartitionTest, PreservesOrder) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(8, 0, ctx);

    auto local = vec.local_view();
    // [3, 1, 4, 1, 5, 9, 2, 6]
    local[0] = 3; local[1] = 1; local[2] = 4; local[3] = 1;
    local[4] = 5; local[5] = 9; local[6] = 2; local[7] = 6;

    auto res = dtl::stable_partition_elements(seq{}, vec,
                                              [](int x) { return x > 3; });

    EXPECT_EQ(res.local_true_count, 4u);  // 4, 5, 9, 6

    // Verify stable ordering: elements > 3 should be in original order
    EXPECT_EQ(local[0], 4);
    EXPECT_EQ(local[1], 5);
    EXPECT_EQ(local[2], 9);
    EXPECT_EQ(local[3], 6);

    // Elements <= 3 should also be in original order
    EXPECT_EQ(local[4], 3);
    EXPECT_EQ(local[5], 1);
    EXPECT_EQ(local[6], 1);
    EXPECT_EQ(local[7], 2);
}

TEST(StablePartitionTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 5; local[1] = 3; local[2] = 8; local[3] = 1;

    auto res = dtl::stable_partition_elements(seq{}, vec,
                                              [](int x) { return x >= 5; });

    EXPECT_EQ(res.local_true_count, 2u);  // 5 and 8
    EXPECT_EQ(local[0], 5);
    EXPECT_EQ(local[1], 8);
}

// =============================================================================
// is_partitioned Tests
// =============================================================================

TEST(IsPartitionedTest, CorrectlyPartitioned) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(6, 0, ctx);

    auto local = vec.local_view();
    // Even first, then odd: [2, 4, 6, 1, 3, 5]
    local[0] = 2; local[1] = 4; local[2] = 6;
    local[3] = 1; local[4] = 3; local[5] = 5;

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x % 2 == 0; });
    EXPECT_TRUE(result);
}

TEST(IsPartitionedTest, NotPartitioned) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 2; local[1] = 3; local[2] = 4; local[3] = 5;

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x % 2 == 0; });
    EXPECT_FALSE(result);
}

TEST(IsPartitionedTest, EmptyIsPartitioned) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x > 0; });
    EXPECT_TRUE(result);
}

TEST(IsPartitionedTest, AllTrueIsPartitioned) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 10, ctx);

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x > 0; });
    EXPECT_TRUE(result);
}

TEST(IsPartitionedTest, AllFalseIsPartitioned) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, -1, ctx);

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x > 0; });
    EXPECT_TRUE(result);
}

TEST(IsPartitionedTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto local = vec.local_view();
    local[0] = 10; local[1] = 20; local[2] = 1; local[3] = 2;

    bool result = dtl::is_partitioned(seq{}, vec,
                                      [](int x) { return x >= 10; });
    EXPECT_TRUE(result);
}

// =============================================================================
// partition_count Tests
// =============================================================================

TEST(PartitionCountTest, CountsCorrectly) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(10, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < 10; ++i) {
        local[i] = static_cast<int>(i);
    }

    size_type count = dtl::partition_count(seq{}, vec,
                                           [](int x) { return x >= 5; });
    EXPECT_EQ(count, 5u);
}

TEST(PartitionCountTest, CountZero) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    size_type count = dtl::partition_count(seq{}, vec,
                                           [](int x) { return x > 100; });
    EXPECT_EQ(count, 0u);
}

TEST(PartitionCountTest, CountAll) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 42, ctx);

    size_type count = dtl::partition_count(seq{}, vec,
                                           [](int x) { return x > 0; });
    EXPECT_EQ(count, 5u);
}

TEST(PartitionCountTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(6, 0, ctx);

    auto local = vec.local_view();
    local[0] = 1; local[1] = -2; local[2] = 3;
    local[3] = -4; local[4] = 5; local[5] = -6;

    size_type count = dtl::partition_count(seq{}, vec,
                                           [](int x) { return x > 0; });
    EXPECT_EQ(count, 3u);
}

// =============================================================================
// partition_result Tests
// =============================================================================

TEST(PartitionResultTest, DefaultConstruction) {
    partition_result result;
    EXPECT_EQ(result.local_true_count, 0u);
    EXPECT_EQ(result.global_true_count, 0u);
    EXPECT_FALSE(result.has_global);
}

TEST(PartitionResultTest, LocalOnly) {
    partition_result result{10, 0, false};
    EXPECT_EQ(result.local_true_count, 10u);
    EXPECT_EQ(result.global_true_count, 0u);
    EXPECT_FALSE(result.has_global);
}

TEST(PartitionResultTest, WithGlobal) {
    partition_result result{10, 40, true};
    EXPECT_EQ(result.local_true_count, 10u);
    EXPECT_EQ(result.global_true_count, 40u);
    EXPECT_TRUE(result.has_global);
}

// =============================================================================
// Multi-Rank Tests
// =============================================================================

TEST(PartitionAlgorithmTest, MultiRankLocal) {
    test_context ctx{2, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i % 3);  // 0, 1, 2, 0, 1, 2, ...
    }

    EXPECT_THROW((void)dtl::partition_elements(seq{}, vec,
                                               [](int x) { return x == 0; }),
                 std::runtime_error);
}

// =============================================================================
// Various Predicate Tests
// =============================================================================

TEST(PartitionAlgorithmTest, GreaterThanPredicate) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(10, 0, ctx);

    auto local = vec.local_view();
    for (size_type i = 0; i < 10; ++i) {
        local[i] = static_cast<int>(i * 10);  // 0, 10, 20, ..., 90
    }

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x > 50; });

    EXPECT_EQ(res.local_true_count, 4u);  // 60, 70, 80, 90
}

TEST(PartitionAlgorithmTest, NegativeValuesPredicate) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(6, 0, ctx);

    auto local = vec.local_view();
    local[0] = -3; local[1] = 5; local[2] = -1;
    local[3] = 7; local[4] = -8; local[5] = 2;

    auto res = dtl::partition_elements(seq{}, vec,
                                       [](int x) { return x < 0; });

    EXPECT_EQ(res.local_true_count, 3u);  // -3, -1, -8
}

}  // namespace dtl::test
