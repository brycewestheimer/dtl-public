// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_stable_sort_global_unit.cpp
/// @brief Unit tests for globally stable sort (Phase 22, Task 22.3)
/// @details Tests that stable_sort with communicator delegates to
///          stable_sort_global and provides global stability. Single-rank
///          tests verify the algorithm logic without requiring real MPI.
/// @note Multi-rank MPI integration tests are in tests/mpi/test_stable_sort_global.cpp

#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/algorithms/sorting/stable_sort_global.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include "mock_single_rank_comm.hpp"

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

/// @brief Element type that tracks original position for stability verification
struct tracked_item {
    int key;
    int original_index;

    bool operator<(const tracked_item& other) const { return key < other.key; }
    bool operator==(const tracked_item& other) const = default;
};

/// @brief Comparator that only compares keys (ignores original_index)
struct compare_by_key {
    bool operator()(const tracked_item& a, const tracked_item& b) const {
        return a.key < b.key;
    }
};

}  // namespace

// =============================================================================
// Stable Sort with Communicator (delegates to stable_sort_global)
// =============================================================================

TEST(StableSortGlobalUnitTest, StableWithEqualElements) {
    distributed_vector<tracked_item> vec(5, test_context{0, 1});

    auto lv = vec.local_view();
    // Two groups of equal keys: key=2 appears at indices 0,2,4
    lv[0] = {2, 0};
    lv[1] = {1, 1};
    lv[2] = {2, 2};
    lv[3] = {1, 3};
    lv[4] = {2, 4};

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, compare_by_key{}, comm);
    EXPECT_TRUE(result.success);

    // After stable sort: sorted by key, equal keys preserve original order
    // Key 1: indices 1, 3
    // Key 2: indices 0, 2, 4
    EXPECT_EQ(lv[0].key, 1);
    EXPECT_EQ(lv[0].original_index, 1);
    EXPECT_EQ(lv[1].key, 1);
    EXPECT_EQ(lv[1].original_index, 3);
    EXPECT_EQ(lv[2].key, 2);
    EXPECT_EQ(lv[2].original_index, 0);
    EXPECT_EQ(lv[3].key, 2);
    EXPECT_EQ(lv[3].original_index, 2);
    EXPECT_EQ(lv[4].key, 2);
    EXPECT_EQ(lv[4].original_index, 4);
}

TEST(StableSortGlobalUnitTest, AlreadySorted) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);

    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 4);
    EXPECT_EQ(lv[4], 5);
}

TEST(StableSortGlobalUnitTest, ReverseSorted) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 4; lv[2] = 3; lv[3] = 2; lv[4] = 1;

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);

    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 4);
    EXPECT_EQ(lv[4], 5);
}

TEST(StableSortGlobalUnitTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);
}

TEST(StableSortGlobalUnitTest, SingleElement) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(1, 42, ctx);

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);

    auto lv = vec.local_view();
    EXPECT_EQ(lv[0], 42);
}

TEST(StableSortGlobalUnitTest, AllSameElements) {
    distributed_vector<tracked_item> vec(4, test_context{0, 1});

    auto lv = vec.local_view();
    lv[0] = {42, 0};
    lv[1] = {42, 1};
    lv[2] = {42, 2};
    lv[3] = {42, 3};

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, compare_by_key{}, comm);
    EXPECT_TRUE(result.success);

    // All equal: original order preserved
    EXPECT_EQ(lv[0].original_index, 0);
    EXPECT_EQ(lv[1].original_index, 1);
    EXPECT_EQ(lv[2].original_index, 2);
    EXPECT_EQ(lv[3].original_index, 3);
}

TEST(StableSortGlobalUnitTest, DescendingComparator) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::greater<>{}, comm);
    EXPECT_TRUE(result.success);

    EXPECT_EQ(lv[0], 5);
    EXPECT_EQ(lv[1], 4);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 2);
    EXPECT_EQ(lv[4], 1);
}

TEST(StableSortGlobalUnitTest, MixedValues) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(8, 0, ctx);

    auto lv = vec.local_view();
    lv[0] = 3; lv[1] = 1; lv[2] = 4; lv[3] = 1;
    lv[4] = 5; lv[5] = 9; lv[6] = 2; lv[7] = 6;

    mock_single_rank_comm comm;
    auto result = dtl::stable_sort(seq{}, vec, std::less<>{}, comm);
    EXPECT_TRUE(result.success);

    EXPECT_TRUE(std::is_sorted(&lv[0], &lv[0] + 8));
}

// =============================================================================
// stable_sort_global standalone tests (single-rank)
// =============================================================================

TEST(StableSortGlobalStandaloneTest, StableLocalSort) {
    distributed_vector<tracked_item> vec(6, test_context{0, 1});

    auto lv = vec.local_view();
    lv[0] = {3, 0};
    lv[1] = {1, 1};
    lv[2] = {3, 2};
    lv[3] = {2, 3};
    lv[4] = {1, 4};
    lv[5] = {3, 5};

    auto res = dtl::stable_sort_global(seq{}, vec, compare_by_key{});
    ASSERT_TRUE(res.has_value());

    // Sorted by key, equal keys preserve original index order
    EXPECT_EQ(lv[0].key, 1); EXPECT_EQ(lv[0].original_index, 1);
    EXPECT_EQ(lv[1].key, 1); EXPECT_EQ(lv[1].original_index, 4);
    EXPECT_EQ(lv[2].key, 2); EXPECT_EQ(lv[2].original_index, 3);
    EXPECT_EQ(lv[3].key, 3); EXPECT_EQ(lv[3].original_index, 0);
    EXPECT_EQ(lv[4].key, 3); EXPECT_EQ(lv[4].original_index, 2);
    EXPECT_EQ(lv[5].key, 3); EXPECT_EQ(lv[5].original_index, 5);
}

TEST(StableSortGlobalStandaloneTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto res = dtl::stable_sort_global(vec);
    ASSERT_TRUE(res.has_value());

    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 4);
    EXPECT_EQ(lv[4], 5);
}

}  // namespace dtl::test
