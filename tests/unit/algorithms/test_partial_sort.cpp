// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_partial_sort.cpp
/// @brief Unit tests for partial_sort algorithm (Phase 06 T08)

#include <dtl/algorithms/sorting/partial_sort.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
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

TEST(PartialSortTest, SortsFirstNElements) {
    distributed_vector<int> vec(8, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 8; lv[1] = 3; lv[2] = 7; lv[3] = 1;
    lv[4] = 5; lv[5] = 2; lv[6] = 6; lv[7] = 4;

    auto res = dtl::partial_sort(vec, 3);
    ASSERT_TRUE(res);

    // First 3 elements should be the 3 smallest (in sorted order)
    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
}

TEST(PartialSortTest, KEqualsZero) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 4; lv[2] = 3; lv[3] = 2; lv[4] = 1;

    auto res = dtl::partial_sort(vec, 0);
    ASSERT_TRUE(res);
    // No elements need to be sorted
}

TEST(PartialSortTest, KEqualsN) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 4; lv[2] = 3; lv[3] = 2; lv[4] = 1;

    auto res = dtl::partial_sort(vec, 5);
    ASSERT_TRUE(res);

    // All elements should be sorted
    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 4);
    EXPECT_EQ(lv[4], 5);
}

TEST(PartialSortTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 42;

    auto res = dtl::partial_sort(vec, 1);
    ASSERT_TRUE(res);
    EXPECT_EQ(lv[0], 42);
}

TEST(PartialSortTest, CustomComparator) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 5; lv[2] = 3; lv[3] = 4; lv[4] = 2;

    // Sort first 2 in descending order
    auto res = dtl::partial_sort(vec, 2, std::greater<>{});
    ASSERT_TRUE(res);
    EXPECT_EQ(lv[0], 5);
    EXPECT_EQ(lv[1], 4);
}

}  // namespace dtl::test
