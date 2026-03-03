// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_composed_view.cpp
/// @brief Unit tests for view composition (composed_view.hpp)
/// @details Phase R7: pipe operator, stride/slice/take/drop adapters

#include <dtl/views/composed_view.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/strided_view.hpp>
#include <dtl/views/subview.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/core/types.hpp>

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
// Pipe Operator: stride
// =============================================================================

TEST(ComposedViewTest, PipeStride2) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,1,2,...,9}

    auto sv = local | stride(2);

    EXPECT_EQ(sv.size(), 5);
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[1], 2);
    EXPECT_EQ(sv[2], 4);
    EXPECT_EQ(sv[3], 6);
    EXPECT_EQ(sv[4], 8);
}

TEST(ComposedViewTest, PipeStride3WithOffset) {
    distributed_vector<int> vec(12, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,1,...,11}

    auto sv = local | stride(3, 1);

    EXPECT_EQ(sv.size(), 4);  // (12 - 1 + 3 - 1) / 3 = 4
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 7);
    EXPECT_EQ(sv[3], 10);
}

TEST(ComposedViewTest, PipeStrideWritable) {
    distributed_vector<int> vec(6, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,1,2,3,4,5}

    auto sv = local | stride(2);
    sv[0] = 100;
    sv[1] = 200;
    sv[2] = 300;

    EXPECT_EQ(local[0], 100);
    EXPECT_EQ(local[1], 1);
    EXPECT_EQ(local[2], 200);
    EXPECT_EQ(local[3], 3);
    EXPECT_EQ(local[4], 300);
    EXPECT_EQ(local[5], 5);
}

// =============================================================================
// Pipe Operator: slice
// =============================================================================

TEST(ComposedViewTest, PipeSlice) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,...,9}

    auto sv = local | slice(3, 4);  // elements [3,7)

    EXPECT_EQ(sv.size(), 4);
    EXPECT_EQ(sv[0], 3);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 5);
    EXPECT_EQ(sv[3], 6);
}

TEST(ComposedViewTest, PipeSliceWritable) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | slice(2, 3);
    sv[0] = 99;
    sv[1] = 88;

    EXPECT_EQ(local[2], 99);
    EXPECT_EQ(local[3], 88);
}

// =============================================================================
// Pipe Operator: take_n
// =============================================================================

TEST(ComposedViewTest, PipeTake) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | take_n(3);

    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[1], 1);
    EXPECT_EQ(sv[2], 2);
}

// =============================================================================
// Pipe Operator: drop_n
// =============================================================================

TEST(ComposedViewTest, PipeDrop) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | drop_n(7);

    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 7);
    EXPECT_EQ(sv[1], 8);
    EXPECT_EQ(sv[2], 9);
}

TEST(ComposedViewTest, PipeDropMoreThanSize) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | drop_n(10);

    EXPECT_EQ(sv.size(), 0);
    EXPECT_TRUE(sv.empty());
}

// =============================================================================
// Composition: Chained Pipes
// =============================================================================

TEST(ComposedViewTest, SliceThenStride) {
    distributed_vector<int> vec(20, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,...,19}

    // First slice [4, 14) then stride by 2
    auto sub = local | slice(4, 10);  // {4,5,6,7,8,9,10,11,12,13}
    auto composed = sub | stride(2);  // {4,6,8,10,12}

    EXPECT_EQ(composed.size(), 5);
    EXPECT_EQ(composed[0], 4);
    EXPECT_EQ(composed[1], 6);
    EXPECT_EQ(composed[2], 8);
    EXPECT_EQ(composed[3], 10);
    EXPECT_EQ(composed[4], 12);
}

TEST(ComposedViewTest, StrideThenSlice) {
    distributed_vector<int> vec(20, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);  // {0,...,19}

    // First stride by 2: {0,2,4,6,8,10,12,14,16,18}
    auto strided = local | stride(2);
    // Then take first 3: {0,2,4}
    auto composed = strided | take_n(3);

    EXPECT_EQ(composed.size(), 3);
    EXPECT_EQ(composed[0], 0);
    EXPECT_EQ(composed[1], 2);
    EXPECT_EQ(composed[2], 4);
}

TEST(ComposedViewTest, TakeThenDrop) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    // Take first 8: {0,...,7}
    auto first8 = local | take_n(8);
    // Drop first 3: {3,4,5,6,7}
    auto result = first8 | drop_n(3);

    EXPECT_EQ(result.size(), 5);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[4], 7);
}

// =============================================================================
// compose_*() Explicit Functions
// =============================================================================

TEST(ComposedViewTest, ComposeStridedFunction) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = compose_strided(local, 3);

    EXPECT_EQ(sv.size(), 4);  // ceil(10/3) = 4
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[1], 3);
    EXPECT_EQ(sv[2], 6);
    EXPECT_EQ(sv[3], 9);
}

TEST(ComposedViewTest, ComposeSubviewFunction) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = compose_subview(local, 2, 5);

    EXPECT_EQ(sv.size(), 5);
    EXPECT_EQ(sv[0], 2);
    EXPECT_EQ(sv[4], 6);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(ComposedViewTest, PipeStrideIteration) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | stride(2);

    std::vector<int> collected;
    for (auto it = sv.begin(); it != sv.end(); ++it) {
        collected.push_back(*it);
    }

    ASSERT_EQ(collected.size(), 5);
    EXPECT_EQ(collected[0], 0);
    EXPECT_EQ(collected[1], 2);
    EXPECT_EQ(collected[2], 4);
    EXPECT_EQ(collected[3], 6);
    EXPECT_EQ(collected[4], 8);
}

TEST(ComposedViewTest, PipeSliceIteration) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | slice(3, 4);

    std::vector<int> collected;
    for (const auto& val : sv) {
        collected.push_back(val);
    }

    ASSERT_EQ(collected.size(), 4);
    EXPECT_EQ(collected[0], 3);
    EXPECT_EQ(collected[1], 4);
    EXPECT_EQ(collected[2], 5);
    EXPECT_EQ(collected[3], 6);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ComposedViewTest, EmptyViewStride) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto local = vec.local_view();

    auto sv = local | stride(2);
    EXPECT_EQ(sv.size(), 0);
    EXPECT_TRUE(sv.empty());
}

TEST(ComposedViewTest, EmptyViewSlice) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto local = vec.local_view();

    auto sv = local | slice(0, 0);
    EXPECT_EQ(sv.size(), 0);
    EXPECT_TRUE(sv.empty());
}

TEST(ComposedViewTest, Stride1IsIdentity) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | stride(1);
    EXPECT_EQ(sv.size(), 5);
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(sv[i], static_cast<int>(i));
    }
}

TEST(ComposedViewTest, TakeZero) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();

    auto sv = local | take_n(0);
    EXPECT_EQ(sv.size(), 0);
}

TEST(ComposedViewTest, DropZero) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = local | drop_n(0);
    EXPECT_EQ(sv.size(), 10);
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[9], 9);
}

// =============================================================================
// std::vector Based Tests (non-DTL view)
// =============================================================================

TEST(ComposedViewTest, StdVectorWithStride) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto lv = make_local_view(data);
    auto sv = lv | stride(4);

    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 0);
    EXPECT_EQ(sv[1], 4);
    EXPECT_EQ(sv[2], 8);
}

TEST(ComposedViewTest, StdVectorWithSlice) {
    std::vector<int> data(12);
    std::iota(data.begin(), data.end(), 0);

    auto lv = make_local_view(data);
    auto sv = lv | slice(5, 3);

    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 5);
    EXPECT_EQ(sv[1], 6);
    EXPECT_EQ(sv[2], 7);
}

}  // namespace dtl::test
