// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_find.cpp
/// @brief Unit tests for find/find_if algorithms (Phase 06 T04/T08)

#include <dtl/algorithms/non_modifying/find.hpp>
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
// find() Tests
// =============================================================================

TEST(FindTest, FindsExistingElement) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 10; lv[1] = 20; lv[2] = 30; lv[3] = 40; lv[4] = 50;

    auto result = dtl::local_find(vec, 30);
    ASSERT_NE(result, lv.end());
    EXPECT_EQ(result - lv.begin(), 2);
    EXPECT_EQ(*result, 30);
}

TEST(FindTest, DoesNotFindNonExistingElement) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 10; lv[1] = 20; lv[2] = 30; lv[3] = 40; lv[4] = 50;

    auto result = dtl::local_find(vec, 99);
    EXPECT_EQ(result, lv.end());
}

TEST(FindTest, FindsFirstOccurrence) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 2; lv[3] = 2; lv[4] = 3;

    auto result = dtl::local_find(vec, 2);
    ASSERT_NE(result, lv.end());
    EXPECT_EQ(result - lv.begin(), 1);  // First occurrence
}

TEST(FindTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto result = dtl::local_find(vec, 42);
    EXPECT_EQ(result, vec.local_view().end());
}

TEST(FindTest, SingleElementFound) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;

    auto result = dtl::local_find(vec, 42);
    ASSERT_NE(result, vec.local_view().end());
    EXPECT_EQ(result - vec.local_view().begin(), 0);
}

TEST(FindTest, SingleElementNotFound) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;

    auto result = dtl::local_find(vec, 0);
    EXPECT_EQ(result, vec.local_view().end());
}

// =============================================================================
// find_if() Tests
// =============================================================================

TEST(FindIfTest, FindsWithPredicate) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    auto result = dtl::local_find_if(vec, [](int x) { return x > 3; });
    ASSERT_NE(result, lv.end());
    EXPECT_EQ(result - lv.begin(), 3);  // First element > 3 is at index 3 (value=4)
    EXPECT_EQ(*result, 4);
}

TEST(FindIfTest, PredicateNeverMatches) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    auto result = dtl::local_find_if(vec, [](int x) { return x > 100; });
    EXPECT_EQ(result, lv.end());
}

TEST(FindIfTest, AllMatch) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3;

    auto result = dtl::local_find_if(vec, [](int x) { return x > 0; });
    ASSERT_NE(result, lv.end());
    EXPECT_EQ(result - lv.begin(), 0);  // First match is index 0
}

TEST(FindIfTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto result = dtl::local_find_if(vec, [](int x) { return x > 0; });
    EXPECT_EQ(result, vec.local_view().end());
}

// =============================================================================
// find_if_not() Tests
// =============================================================================

TEST(FindIfNotTest, FindsFirstNonMatch) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 4; lv[2] = 6; lv[3] = 7; lv[4] = 8;

    auto result = dtl::local_find_if(vec, [](int x) { return x % 2 != 0; });
    ASSERT_NE(result, lv.end());
    EXPECT_EQ(*result, 7);  // First odd number
}

TEST(FindIfNotTest, AllMatch) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 2; lv[1] = 4; lv[2] = 6;

    auto result = dtl::local_find_if(vec, [](int x) { return x % 2 != 0; });
    EXPECT_EQ(result, lv.end());  // All are even
}

}  // namespace dtl::test
