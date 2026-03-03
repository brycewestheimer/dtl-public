// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_unique_distributed.cpp
/// @brief Unit tests for distributed unique with communicator (Phase 22, Task 22.2)
/// @details Tests the comm-based unique() that handles boundary duplicates
///          between adjacent ranks. Single-rank tests verify algorithm correctness
///          through mock communicator.

#include <dtl/algorithms/sorting/unique.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include "mock_single_rank_comm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
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
// Distributed Unique with Communicator Tests (single-rank)
// =============================================================================

TEST(DistributedUniqueTest, RemovesConsecutiveDuplicates) {
    distributed_vector<int> vec(7, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 1; lv[2] = 2; lv[3] = 3;
    lv[4] = 3; lv[5] = 3; lv[6] = 4;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 4u);
    EXPECT_EQ(res.value().removed_count, 3u);

    auto compacted = vec.local_view();
    EXPECT_EQ(compacted[0], 1);
    EXPECT_EQ(compacted[1], 2);
    EXPECT_EQ(compacted[2], 3);
    EXPECT_EQ(compacted[3], 4);
}

TEST(DistributedUniqueTest, AllUnique) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 5u);
    EXPECT_EQ(res.value().removed_count, 0u);
}

TEST(DistributedUniqueTest, AllSame) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = 42;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 1u);
    EXPECT_EQ(res.value().removed_count, 4u);
    auto compacted = vec.local_view();
    EXPECT_EQ(compacted[0], 42);
}

TEST(DistributedUniqueTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 7;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 1u);
    EXPECT_EQ(res.value().removed_count, 0u);
}

TEST(DistributedUniqueTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 0u);
    EXPECT_EQ(res.value().removed_count, 0u);
}

TEST(DistributedUniqueTest, CustomPredicate) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto lv = vec.local_view();
    // Consider consecutive elements equal if they differ by at most 1
    lv[0] = 1; lv[1] = 2; lv[2] = 5; lv[3] = 6; lv[4] = 6; lv[5] = 10;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec,
        [](int a, int b) { return std::abs(a - b) <= 1; }, comm);
    ASSERT_TRUE(res);
    // After unique: 1, 5, 10
    EXPECT_EQ(res.value().new_size, 3u);
}

TEST(DistributedUniqueTest, SortedInput) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 1; lv[2] = 2; lv[3] = 2; lv[4] = 2;
    lv[5] = 3; lv[6] = 4; lv[7] = 4; lv[8] = 5; lv[9] = 5;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 5u);
    EXPECT_EQ(res.value().removed_count, 5u);
    auto compacted = vec.local_view();
    EXPECT_EQ(compacted[0], 1);
    EXPECT_EQ(compacted[1], 2);
    EXPECT_EQ(compacted[2], 3);
    EXPECT_EQ(compacted[3], 4);
    EXPECT_EQ(compacted[4], 5);
}

TEST(DistributedUniqueTest, TwoElements) {
    distributed_vector<int> vec(2, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 5;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 1u);
    EXPECT_EQ(res.value().removed_count, 1u);
}

TEST(DistributedUniqueTest, TwoDistinctElements) {
    distributed_vector<int> vec(2, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 10;

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 2u);
    EXPECT_EQ(res.value().removed_count, 0u);
}

TEST(DistributedUniqueTest, LargeAlternatingDuplicates) {
    // Large dataset with alternating pairs
    distributed_vector<int> vec(100, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 100; ++i) {
        lv[i] = static_cast<int>(i / 2);  // 0,0,1,1,2,2,...,49,49
    }

    mock_single_rank_comm comm;
    auto res = dtl::unique(seq{}, vec, std::equal_to<>{}, comm);
    ASSERT_TRUE(res);
    EXPECT_EQ(res.value().new_size, 50u);
    EXPECT_EQ(res.value().removed_count, 50u);
}

// =============================================================================
// Count Duplicates with Communicator Tests
// =============================================================================

TEST(DistributedCountDuplicatesTest, CountsCorrectly) {
    distributed_vector<int> vec(7, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 1; lv[2] = 2; lv[3] = 3;
    lv[4] = 3; lv[5] = 3; lv[6] = 4;

    mock_single_rank_comm comm;
    auto count = dtl::count_duplicates(seq{}, vec, std::equal_to<>{}, comm);
    EXPECT_EQ(count, 3u);  // 1-1, 3-3, 3-3
}

TEST(DistributedCountDuplicatesTest, NoDuplicates) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    mock_single_rank_comm comm;
    auto count = dtl::count_duplicates(seq{}, vec, std::equal_to<>{}, comm);
    EXPECT_EQ(count, 0u);
}

// =============================================================================
// Has Duplicates with Communicator Tests
// =============================================================================

TEST(DistributedHasDuplicatesTest, DetectsDuplicates) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 2; lv[3] = 3; lv[4] = 4;

    mock_single_rank_comm comm;
    EXPECT_TRUE(dtl::has_duplicates(seq{}, vec, std::equal_to<>{}, comm));
}

TEST(DistributedHasDuplicatesTest, NoDuplicates) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    mock_single_rank_comm comm;
    EXPECT_FALSE(dtl::has_duplicates(seq{}, vec, std::equal_to<>{}, comm));
}

}  // namespace dtl::test
