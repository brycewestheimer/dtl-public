// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nth_element.cpp
/// @brief Unit tests for nth_element algorithm (Phase 06 T08)

#include <dtl/algorithms/sorting/nth_element.hpp>
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

TEST(DistributedNthElementTest, FindsMedian) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto res = dtl::nth_element(vec, 2);  // Find median (index 2)
    ASSERT_TRUE(res.valid);
    EXPECT_EQ(res.value, 3);  // Third smallest = 3
}

TEST(DistributedNthElementTest, FindsMinimum) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto res = dtl::nth_element(vec, 0);  // Find minimum
    ASSERT_TRUE(res.valid);
    EXPECT_EQ(res.value, 1);
}

TEST(DistributedNthElementTest, FindsMaximum) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 5; lv[1] = 3; lv[2] = 1; lv[3] = 4; lv[4] = 2;

    auto res = dtl::nth_element(vec, 4);  // Find maximum
    ASSERT_TRUE(res.valid);
    EXPECT_EQ(res.value, 5);
}

TEST(DistributedNthElementTest, PartitionProperty) {
    distributed_vector<int> vec(7, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 7; lv[1] = 2; lv[2] = 5; lv[3] = 1;
    lv[4] = 6; lv[5] = 3; lv[6] = 4;

    auto res = dtl::nth_element(vec, 3);
    ASSERT_TRUE(res.valid);

    // After nth_element, elements before n should be <= nth, elements after >= nth
    int nth_val = lv[3];
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_LE(lv[i], nth_val);
    }
    for (size_t i = 4; i < 7; ++i) {
        EXPECT_GE(lv[i], nth_val);
    }
}

TEST(DistributedNthElementTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 42;

    auto res = dtl::nth_element(vec, 0);
    ASSERT_TRUE(res.valid);
    EXPECT_EQ(res.value, 42);
}

TEST(DistributedNthElementTest, InvalidIndex) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = static_cast<int>(i);

    // Negative index should return invalid
    auto res = dtl::nth_element(vec, -1);
    EXPECT_FALSE(res.valid);
}

}  // namespace dtl::test
