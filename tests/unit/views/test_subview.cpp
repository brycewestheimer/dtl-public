// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_subview.cpp
/// @brief Unit tests for subview
/// @details Phase 08, Task 05

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/subview.hpp>

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

TEST(SubviewTest, OffsetAndSize) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5}

    auto sv = make_subview(local, 1, 3);  // offset=1, count=3 => {2, 3, 4}
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 2);
    EXPECT_EQ(sv[1], 3);
    EXPECT_EQ(sv[2], 4);
}

TEST(SubviewTest, FromStart) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 0, 3);  // {1, 2, 3}
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[2], 3);
}

TEST(SubviewTest, ToEnd) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 3, 2);  // {4, 5}
    EXPECT_EQ(sv.size(), 2);
    EXPECT_EQ(sv[0], 4);
    EXPECT_EQ(sv[1], 5);
}

TEST(SubviewTest, FullRange) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 0, 5);  // All elements
    EXPECT_EQ(sv.size(), 5);
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(sv[i], static_cast<int>(i + 1));
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SubviewTest, EmptySubview) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();

    auto sv = make_subview(local, 2, 0);  // Empty
    EXPECT_TRUE(sv.empty());
    EXPECT_EQ(sv.size(), 0);
    // begin() == end() for empty subviews
    EXPECT_EQ(sv.begin(), sv.end());
}

TEST(SubviewTest, SingleElement) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 2, 1);  // {3}
    EXPECT_EQ(sv.size(), 1);
    EXPECT_EQ(sv[0], 3);
    EXPECT_EQ(sv.front(), 3);
    EXPECT_EQ(sv.back(), 3);
}

// =============================================================================
// Write Through Tests
// =============================================================================

TEST(SubviewTest, WriteThroughModifiesUnderlying) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);  // {1, 2, 3, 4, 5}

    auto sv = make_subview(local, 1, 3);
    sv[0] = 20;
    sv[1] = 30;
    sv[2] = 40;

    EXPECT_EQ(local[0], 1);   // untouched
    EXPECT_EQ(local[1], 20);  // modified
    EXPECT_EQ(local[2], 30);  // modified
    EXPECT_EQ(local[3], 40);  // modified
    EXPECT_EQ(local[4], 5);   // untouched
}

// =============================================================================
// Iteration Tests
// =============================================================================

TEST(SubviewTest, RangeBasedFor) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 1, 3);
    std::vector<int> result;
    for (auto& val : sv) {
        result.push_back(val);
    }
    ASSERT_EQ(result.size(), 3u);
    EXPECT_EQ(result[0], 2);
    EXPECT_EQ(result[1], 3);
    EXPECT_EQ(result[2], 4);
}

TEST(SubviewTest, ConstIteration) {
    distributed_vector<int> vec(5, 42, test_context{0, 1});
    auto local = vec.local_view();

    const auto sv = make_subview(local, 0, 5);
    for (const auto& val : sv) {
        EXPECT_EQ(val, 42);
    }
}

// =============================================================================
// Data Pointer Tests
// =============================================================================

TEST(SubviewTest, DataPointer) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 2, 3);
    EXPECT_EQ(*sv.data(), 3);  // data() points to first element of subview
    EXPECT_EQ(sv.data()[0], 3);
    EXPECT_EQ(sv.data()[1], 4);
    EXPECT_EQ(sv.data()[2], 5);
}

// =============================================================================
// Front/Back Tests
// =============================================================================

TEST(SubviewTest, FrontAndBack) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = make_subview(local, 1, 3);  // {2, 3, 4}
    EXPECT_EQ(sv.front(), 2);
    EXPECT_EQ(sv.back(), 4);
}

// =============================================================================
// Offset Tests
// =============================================================================

TEST(SubviewTest, OffsetProperty) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();

    auto sv = make_subview(local, 3, 5);
    EXPECT_EQ(sv.offset(), 3);
}

// =============================================================================
// Subrange of Subview Tests
// =============================================================================

TEST(SubviewTest, SubrangeOfSubview) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv1 = make_subview(local, 2, 6);  // {2,3,4,5,6,7}
    auto sv2 = sv1.subrange(1, 3);         // {3,4,5}
    EXPECT_EQ(sv2.size(), 3);
    EXPECT_EQ(sv2[0], 3);
    EXPECT_EQ(sv2[1], 4);
    EXPECT_EQ(sv2[2], 5);
}

// =============================================================================
// Factory Functions
// =============================================================================

TEST(SubviewTest, TakeFunction) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = take(local, 3);  // First 3 elements: {1, 2, 3}
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 1);
    EXPECT_EQ(sv[2], 3);
}

TEST(SubviewTest, DropFunction) {
    distributed_vector<int> vec(5, test_context{0, 1});
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 1);

    auto sv = drop(local, 2);  // Skip first 2: {3, 4, 5}
    EXPECT_EQ(sv.size(), 3);
    EXPECT_EQ(sv[0], 3);
    EXPECT_EQ(sv[2], 5);
}

}  // namespace dtl::test
