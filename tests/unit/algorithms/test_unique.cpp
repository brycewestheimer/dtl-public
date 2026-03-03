// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_unique.cpp
/// @brief Unit tests for unique algorithm (Phase 06 T08)

#include <dtl/algorithms/sorting/unique.hpp>
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

TEST(UniqueTest, RemovesConsecutiveDuplicates) {
    distributed_vector<int> vec(7, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 1; lv[2] = 2; lv[3] = 3;
    lv[4] = 3; lv[5] = 3; lv[6] = 4;

    const auto removed = dtl::local_unique(vec);
    EXPECT_EQ(removed, 3u);
    EXPECT_EQ(lv.size() - removed, 4u);

    EXPECT_EQ(lv[0], 1);
    EXPECT_EQ(lv[1], 2);
    EXPECT_EQ(lv[2], 3);
    EXPECT_EQ(lv[3], 4);
}

TEST(UniqueTest, AllUnique) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 1; lv[1] = 2; lv[2] = 3; lv[3] = 4; lv[4] = 5;

    const auto removed = dtl::local_unique(vec);
    EXPECT_EQ(removed, 0u);
    EXPECT_EQ(lv.size() - removed, 5u);
}

TEST(UniqueTest, AllSame) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto lv = vec.local_view();
    for (size_type i = 0; i < 5; ++i) lv[i] = 42;

    const auto removed = dtl::local_unique(vec);
    EXPECT_EQ(removed, 4u);
    EXPECT_EQ(lv.size() - removed, 1u);
    EXPECT_EQ(lv[0], 42);
}

TEST(UniqueTest, SingleElement) {
    distributed_vector<int> vec(1, test_context{0, 1});
    auto lv = vec.local_view();
    lv[0] = 7;

    const auto removed = dtl::local_unique(vec);
    EXPECT_EQ(removed, 0u);
    EXPECT_EQ(lv.size() - removed, 1u);
}

TEST(UniqueTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});

    const auto removed = dtl::local_unique(vec);
    EXPECT_EQ(removed, 0u);
}

TEST(UniqueTest, CustomPredicate) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto lv = vec.local_view();
    // Consider consecutive elements equal if they differ by at most 1
    lv[0] = 1; lv[1] = 2; lv[2] = 5; lv[3] = 6; lv[4] = 6; lv[5] = 10;

    const auto removed =
        dtl::local_unique(vec, [](int a, int b) { return std::abs(a - b) <= 1; });
    // After unique: 1, 5, 10
    EXPECT_EQ(lv.size() - removed, 3u);
}

}  // namespace dtl::test
