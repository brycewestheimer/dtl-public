// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_strided_view.cpp
/// @brief Unit tests for strided_view
/// @details Phase 08, Task 04

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/strided_view.hpp>

#include <gtest/gtest.h>

#include <numeric>

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
// Construction Tests
// =============================================================================

TEST(StridedViewTest, ConstructStride2) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5, 6}

    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv.size(), 3);
}

TEST(StridedViewTest, ConstructStride1) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_strided_view(local, 1);
    EXPECT_EQ(sv.size(), 6);  // stride=1 is identity
}

TEST(StridedViewTest, ConstructStride3) {
    distributed_vector<int> vec(9, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1,2,3,4,5,6,7,8,9}

    auto sv = make_strided_view(local, 3);
    EXPECT_EQ(sv.size(), 3);
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(StridedViewTest, Stride2YieldsCorrectValues) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5, 6}

    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 3);
    EXPECT_EQ(sv[2], 5);
}

TEST(StridedViewTest, Stride3YieldsCorrectValues) {
    distributed_vector<int> vec(9, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1,2,3,4,5,6,7,8,9}

    auto sv = make_strided_view(local, 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 7);
}

TEST(StridedViewTest, FrontAndBack) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5, 6}

    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv.front(), 1);
    EXPECT_EQ(sv.back(), 5);
}

// =============================================================================
// Iteration Tests
// =============================================================================

TEST(StridedViewTest, IterateStride2) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_strided_view(local, 2);
    std::vector<int> result;
    for (auto it = sv.begin(); it != sv.end(); ++it) {
        result.push_back(*it);
    }
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 5);
}

TEST(StridedViewTest, RangeBasedFor) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_strided_view(local, 2);
    std::vector<int> result;
    for (auto& val : sv) {
        result.push_back(val);
    }
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 5);
}

// =============================================================================
// Write Through Tests
// =============================================================================

TEST(StridedViewTest, WriteThroughModifiesUnderlying) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5, 6}

    auto sv = make_strided_view(local, 2);
    sv[0] = 100;
    sv[1] = 200;
    sv[2] = 300;

    EXPECT_EQ(local[0], 100);  // index 0
    EXPECT_EQ(local[1], 2);    // untouched
    EXPECT_EQ(local[2], 200);  // index 2
    EXPECT_EQ(local[3], 4);    // untouched
    EXPECT_EQ(local[4], 300);  // index 4
    EXPECT_EQ(local[5], 6);    // untouched
}

// =============================================================================
// Size Computation Tests
// =============================================================================

TEST(StridedViewTest, SizeExactDivision) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv.size(), 3);  // 6 / 2 = 3
}

TEST(StridedViewTest, SizeNotExactDivision) {
    distributed_vector<int> vec(7, test_context{0, 1});
    auto local = vec.local_view();
    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv.size(), 4);  // ceil(7 / 2) = 4
}

TEST(StridedViewTest, SizeStrideLargerThanContainer) {
    distributed_vector<int> vec(3, test_context{0, 1});
    auto local = vec.local_view();
    auto sv = make_strided_view(local, 10);
    EXPECT_EQ(sv.size(), 1);  // ceil(3 / 10) = 1, just the first element
}

// =============================================================================
// Offset Tests
// =============================================================================

TEST(StridedViewTest, OffsetByOne) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5, 6}

    auto sv = make_strided_view(local, 2, 1);  // offset=1
    // Should yield: {2, 4, 6}
    EXPECT_EQ(sv[0], 2);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 6);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(StridedViewTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto local = vec.local_view();
    auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv.size(), 0);
    EXPECT_TRUE(sv.empty());
}

TEST(StridedViewTest, StrideAndOffsetProperties) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    auto sv = make_strided_view(local, 3, 1);
    EXPECT_EQ(sv.stride(), 3);
    EXPECT_EQ(sv.offset(), 1);
}

// =============================================================================
// Const Correctness
// =============================================================================

TEST(StridedViewTest, ConstStridedView) {
    distributed_vector<int> vec(6, 42, test_context{0, 1});
    auto local = vec.local_view();

    const auto sv = make_strided_view(local, 2);
    EXPECT_EQ(sv[0], 42);
    EXPECT_EQ(sv.size(), 3);
}

}  // namespace dtl::test
