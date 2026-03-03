// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_replace.cpp
/// @brief Unit tests for replace algorithm (Phase 06 T08)

#include <dtl/algorithms/modifying/replace.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

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
// Replace Tests
// =============================================================================

TEST(ReplaceTest, ReplacesMatchingElements) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 2; lv[4] = 5;

    auto count = dtl::local_replace(vec, 2, 99);
    EXPECT_EQ(count, 2u);

    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 99);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 99);
    EXPECT_EQ(lv[4], 5);
}

TEST(ReplaceTest, NoMatchReplacesNothing) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = static_cast<int>(i);

    auto count = dtl::local_replace(vec, 42, 99);
    EXPECT_EQ(count, 0u);
}

TEST(ReplaceTest, AllMatchReplaceAll) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 4; ++i) lv[i] = 7;

    auto count = dtl::local_replace(vec, 7, 0);
    EXPECT_EQ(count, 4u);
    for (size_type i = 0; i < 4; ++i) EXPECT_EQ(lv[i], 0);
}

TEST(ReplaceTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto count = dtl::local_replace(vec, 1, 2);
    EXPECT_EQ(count, 0u);
}

// =============================================================================
// Replace_if Tests
// =============================================================================

TEST(ReplaceIfTest, ReplacesWithPredicate) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = -1; lv[1] = 2; lv[2] = -3; lv[3] = 4; lv[4] = -5;

    auto count = dtl::local_replace_if(vec, [](int x) { return x < 0; }, 0);
    EXPECT_EQ(count, 3u);
    EXPECT_EQ(lv[0], 0);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 0);
    EXPECT_EQ(lv[3], 4);
    EXPECT_EQ(lv[4], 0);
}

TEST(ReplaceIfTest, NoneMatchPredicate) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3;

    auto count = dtl::local_replace_if(vec, [](int x) { return x > 10; }, 0);
    EXPECT_EQ(count, 0u);
}

}  // namespace dtl::test
