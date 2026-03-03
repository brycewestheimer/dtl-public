// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_minmax.cpp
/// @brief Unit tests for minmax algorithms (Phase 06 T08)

#include <dtl/algorithms/reductions/minmax.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

#include <limits>
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
// min_element Tests
// =============================================================================

TEST(MinElementTest, FindsMinimum) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto result = dtl::min_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 1);
}

TEST(MinElementTest, MinAtBeginning) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 0; lv[1] = 1; lv[2] = 2; lv[3] = 3; lv[4] = 4;

    auto result = dtl::min_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 0);
}

TEST(MinElementTest, MinAtEnd) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 4; lv[1] = 3; lv[2] = 2; lv[3] = 1; lv[4] = 0;

    auto result = dtl::min_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 0);
}

TEST(MinElementTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = 42;

    auto result = dtl::min_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 42);
}

TEST(MinElementTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto result = dtl::min_element(vec);
    EXPECT_FALSE(result.valid);
}

TEST(MinElementTest, NegativeValues) {
    distributed_vector<int> vec(4, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = -1; lv[1] = -5; lv[2] = -3; lv[3] = -2;

    auto result = dtl::min_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, -5);
}

// =============================================================================
// max_element Tests
// =============================================================================

TEST(MaxElementTest, FindsMaximum) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto result = dtl::max_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 5);
}

TEST(MaxElementTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    vec.local_view()[0] = -100;

    auto result = dtl::max_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, -100);
}

TEST(MaxElementTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto result = dtl::max_element(vec);
    EXPECT_FALSE(result.valid);
}

TEST(MaxElementTest, AllSameValues) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = 7;

    auto result = dtl::max_element(vec);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.value, 7);
}

// =============================================================================
// minmax_element Tests
// =============================================================================

TEST(MinmaxElementTest, FindsMinAndMax) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto min_res = dtl::min_element(vec);
    auto max_res = dtl::max_element(vec);

    ASSERT_TRUE(min_res.valid);
    ASSERT_TRUE(max_res.valid);
    EXPECT_EQ(min_res.value, 1);
    EXPECT_EQ(max_res.value, 5);
}

}  // namespace dtl::test
