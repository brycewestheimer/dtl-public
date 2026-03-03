// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_iota.cpp
/// @brief Unit tests for distributed iota algorithm (R6.3)

#include <dtl/algorithms/modifying/iota.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/containers/distributed_vector.hpp>

#include <gtest/gtest.h>

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
// Distributed Iota Tests (Global Offset-Aware)
// =============================================================================

TEST(DistributedIotaTest, SingleRankBasic) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 1);
    EXPECT_EQ(local[2], 2);
    EXPECT_EQ(local[3], 3);
    EXPECT_EQ(local[4], 4);
}

TEST(DistributedIotaTest, SingleRankWithStartValue) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 10);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 10);
    EXPECT_EQ(local[1], 11);
    EXPECT_EQ(local[2], 12);
    EXPECT_EQ(local[3], 13);
    EXPECT_EQ(local[4], 14);
}

TEST(DistributedIotaTest, SingleRankNegativeStart) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, -2);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], -2);
    EXPECT_EQ(local[1], -1);
    EXPECT_EQ(local[2], 0);
    EXPECT_EQ(local[3], 1);
    EXPECT_EQ(local[4], 2);
}

TEST(DistributedIotaTest, MultiRankRank0) {
    // Rank 0 of 4, 100 global elements = 25 local
    // global_offset for rank 0 = 0
    test_context ctx{0, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 25u);
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(i));
    }
}

TEST(DistributedIotaTest, MultiRankRank1) {
    // Rank 1 of 4, 100 global elements = 25 local
    // global_offset for rank 1 = 25
    test_context ctx{1, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 25u);
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(25 + i));
    }
}

TEST(DistributedIotaTest, MultiRankRank3) {
    // Rank 3 of 4, 100 global elements = 25 local
    // global_offset for rank 3 = 75
    test_context ctx{3, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 25u);
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(75 + i));
    }
}

TEST(DistributedIotaTest, MultiRankWithStartValue) {
    // Rank 2 of 4, start = 1000
    // global_offset for rank 2 = 50
    test_context ctx{2, 4};
    distributed_vector<int> vec(100, 0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 1000);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(1000 + 50 + i));
    }
}

TEST(DistributedIotaTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto res = dtl::distributed_iota(vec, 42);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 42);
    EXPECT_EQ(local[4], 46);
}

TEST(DistributedIotaTest, EmptyContainer) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 0);
    ASSERT_TRUE(res.has_value());
}

// =============================================================================
// Local Iota Tests (No Global Offset)
// =============================================================================

TEST(LocalIotaTest, BasicLocalIota) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    size_type count = dtl::local_iota(seq{}, vec, 0);
    EXPECT_EQ(count, 5u);

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 1);
    EXPECT_EQ(local[2], 2);
    EXPECT_EQ(local[3], 3);
    EXPECT_EQ(local[4], 4);
}

TEST(LocalIotaTest, WithStartValue) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    size_type count = dtl::local_iota(seq{}, vec, 100);
    EXPECT_EQ(count, 3u);

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 100);
    EXPECT_EQ(local[1], 101);
    EXPECT_EQ(local[2], 102);
}

TEST(LocalIotaTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    size_type count = dtl::local_iota(vec, 10);
    EXPECT_EQ(count, 5u);

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 10);
    EXPECT_EQ(local[4], 14);
}

TEST(LocalIotaTest, MultiRankLocalStart) {
    // Even for rank 2, local_iota starts from the given start value
    // (not adjusted by global offset)
    test_context ctx{2, 4};
    distributed_vector<int> vec(100, 0, ctx);

    size_type count = dtl::local_iota(seq{}, vec, 0);
    EXPECT_EQ(count, 25u);

    auto local = vec.local_view();
    // Should start from 0, not from global offset
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 1);
}

// =============================================================================
// Iota with Step Tests
// =============================================================================

TEST(IotaStepTest, BasicStep) {
    test_context ctx{0, 1};
    distributed_vector<double> vec(5, 0.0, ctx);

    auto res = dtl::iota_step(seq{}, vec, 0.0, 0.5);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_DOUBLE_EQ(local[0], 0.0);
    EXPECT_DOUBLE_EQ(local[1], 0.5);
    EXPECT_DOUBLE_EQ(local[2], 1.0);
    EXPECT_DOUBLE_EQ(local[3], 1.5);
    EXPECT_DOUBLE_EQ(local[4], 2.0);
}

TEST(IotaStepTest, IntegerStep) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(5, 0, ctx);

    auto res = dtl::iota_step(seq{}, vec, 10, 3);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 10);
    EXPECT_EQ(local[1], 13);
    EXPECT_EQ(local[2], 16);
    EXPECT_EQ(local[3], 19);
    EXPECT_EQ(local[4], 22);
}

TEST(IotaStepTest, NegativeStep) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(4, 0, ctx);

    auto res = dtl::iota_step(seq{}, vec, 100, -10);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 100);
    EXPECT_EQ(local[1], 90);
    EXPECT_EQ(local[2], 80);
    EXPECT_EQ(local[3], 70);
}

TEST(IotaStepTest, DefaultExecution) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(3, 0, ctx);

    auto res = dtl::iota_step(vec, 0, 2);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[1], 2);
    EXPECT_EQ(local[2], 4);
}

TEST(IotaStepTest, MultiRankStep) {
    // Rank 1 of 2, 10 global elements = 5 local per rank
    // global_offset for rank 1 = 5
    // start=0, step=2: global values would be [0,2,4,6,8,10,12,14,16,18]
    // rank 1 local should be [10,12,14,16,18]
    test_context ctx{1, 2};
    distributed_vector<int> vec(10, 0, ctx);

    auto res = dtl::iota_step(seq{}, vec, 0, 2);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 5u);
    EXPECT_EQ(local[0], 10);  // (0 + 5*2) + 0*2
    EXPECT_EQ(local[1], 12);
    EXPECT_EQ(local[2], 14);
    EXPECT_EQ(local[3], 16);
    EXPECT_EQ(local[4], 18);
}

// =============================================================================
// Double Type Tests
// =============================================================================

TEST(DistributedIotaTest, DoubleValues) {
    test_context ctx{0, 1};
    distributed_vector<double> vec(5, 0.0, ctx);

    auto res = dtl::distributed_iota(seq{}, vec, 1.5);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    EXPECT_DOUBLE_EQ(local[0], 1.5);
    EXPECT_DOUBLE_EQ(local[1], 2.5);
    EXPECT_DOUBLE_EQ(local[2], 3.5);
    EXPECT_DOUBLE_EQ(local[3], 4.5);
    EXPECT_DOUBLE_EQ(local[4], 5.5);
}

}  // namespace dtl::test
