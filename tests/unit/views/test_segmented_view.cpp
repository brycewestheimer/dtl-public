// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_segmented_view.cpp
/// @brief Unit tests for segmented_view
/// @details Phase 08, Task 03: segmented_view is the primary iteration substrate
///          for distributed algorithms. This file provides comprehensive coverage.

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/distributed_tensor.hpp>
#include <dtl/views/segmented_view.hpp>

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

TEST(SegmentedViewTest, ConstructFromVector) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto sv = vec.segmented_view();
    EXPECT_EQ(sv.num_segments(), 1);
    EXPECT_EQ(sv.total_size(), 100);
}

TEST(SegmentedViewTest, ConstructFromArray) {
    distributed_array<int, 100> arr(test_context{0, 1});
    auto sv = arr.segmented_view();
    EXPECT_EQ(sv.num_segments(), 1);
    EXPECT_EQ(sv.total_size(), 100);
}

TEST(SegmentedViewTest, ConstructFromTensor) {
    distributed_tensor<int, 2> tensor({10, 10}, test_context{0, 1});
    auto sv = tensor.segmented_view();
    EXPECT_EQ(sv.num_segments(), 1);
    EXPECT_EQ(sv.total_size(), 100);
}

TEST(SegmentedViewTest, FactoryFunction) {
    distributed_vector<int> vec(50, test_context{0, 1});
    auto sv = make_segmented_view(vec);
    EXPECT_EQ(sv.num_segments(), 1);
    EXPECT_EQ(sv.total_size(), 50);
}

// =============================================================================
// Segment Iteration Tests
// =============================================================================

TEST(SegmentedViewTest, SingleRankOneSeg) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto sv = vec.segmented_view();

    int count = 0;
    for (auto seg : sv) {
        ++count;
        EXPECT_TRUE(seg.is_local());
        EXPECT_FALSE(seg.is_remote());
        EXPECT_EQ(seg.size(), 100);
    }
    EXPECT_EQ(count, 1);
}

TEST(SegmentedViewTest, MultiRankSegmentCount) {
    // Simulate 4-rank distributed vector, we are rank 1
    distributed_vector<int> vec(100, test_context{1, 4});
    auto sv = vec.segmented_view();

    EXPECT_EQ(sv.num_segments(), 4);

    int count = 0;
    int local_count = 0;
    int remote_count = 0;
    for (auto seg : sv) {
        ++count;
        if (seg.is_local()) {
            ++local_count;
        } else {
            ++remote_count;
        }
    }
    EXPECT_EQ(count, 4);
    EXPECT_EQ(local_count, 1);  // Only our rank's segment is local
    EXPECT_EQ(remote_count, 3);
}

TEST(SegmentedViewTest, LocalSegmentSizeCorrect) {
    distributed_vector<int> vec(100, test_context{1, 4});
    auto sv = vec.segmented_view();

    auto local_seg = sv.local_segment();
    EXPECT_TRUE(local_seg.is_local());
    EXPECT_EQ(local_seg.size(), 25);  // 100 / 4 = 25
}

TEST(SegmentedViewTest, SegmentGlobalOffset) {
    distributed_vector<int> vec(100, test_context{0, 4});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    EXPECT_EQ(seg.global_offset, 0);  // Rank 0 starts at offset 0
}

TEST(SegmentedViewTest, SegmentGlobalOffsetRank2) {
    distributed_vector<int> vec(100, test_context{2, 4});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    EXPECT_EQ(seg.global_offset, 50);  // Rank 2 starts at offset 50
}

// =============================================================================
// Element Access Tests
// =============================================================================

TEST(SegmentedViewTest, LocalSegmentElementAccess) {
    distributed_vector<int> vec(10, 42, test_context{0, 1});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    for (size_type i = 0; i < seg.size(); ++i) {
        EXPECT_EQ(seg[i], 42);
    }
}

TEST(SegmentedViewTest, LocalSegmentIteration) {
    distributed_vector<int> vec(10, test_context{0, 1});
    // Fill with values
    auto local = vec.local_view();
    std::iota(local.begin(), local.end(), 0);

    auto sv = vec.segmented_view();
    auto seg = sv.local_segment();

    int expected = 0;
    for (auto& elem : seg) {
        EXPECT_EQ(elem, expected++);
    }
    EXPECT_EQ(expected, 10);
}

TEST(SegmentedViewTest, WriteViaSeg) {
    distributed_vector<int> vec(10, test_context{0, 1});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    seg[0] = 99;
    EXPECT_EQ(vec.local(0), 99);
}

TEST(SegmentedViewTest, SegmentToGlobalIndex) {
    distributed_vector<int> vec(100, test_context{2, 4});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    // Rank 2's first element has local index 0, global index 50
    EXPECT_EQ(seg.to_global(0), 50);
    EXPECT_EQ(seg.to_global(24), 74);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SegmentedViewTest, EmptyContainer) {
    distributed_vector<int> vec(0, test_context{0, 1});
    auto sv = vec.segmented_view();

    EXPECT_EQ(sv.num_segments(), 1);
    EXPECT_EQ(sv.total_size(), 0);

    auto seg = sv.local_segment();
    EXPECT_TRUE(seg.empty());
    EXPECT_EQ(seg.size(), 0);
}

TEST(SegmentedViewTest, SingleElement) {
    distributed_vector<int> vec(1, 42, test_context{0, 1});
    auto sv = vec.segmented_view();

    auto seg = sv.local_segment();
    EXPECT_EQ(seg.size(), 1);
    EXPECT_EQ(seg[0], 42);
}

// =============================================================================
// Offset Cache Tests
// =============================================================================

TEST(SegmentedViewTest, OffsetForRank) {
    distributed_vector<int> vec(100, test_context{0, 4});
    auto sv = vec.segmented_view();

    EXPECT_EQ(sv.offset_for_rank(0), 0);
    EXPECT_EQ(sv.offset_for_rank(1), 25);
    EXPECT_EQ(sv.offset_for_rank(2), 50);
    EXPECT_EQ(sv.offset_for_rank(3), 75);
}

TEST(SegmentedViewTest, SegmentForRank) {
    distributed_vector<int> vec(100, test_context{0, 4});
    auto sv = vec.segmented_view();

    auto seg0 = sv.segment_for_rank(0);
    EXPECT_TRUE(seg0.is_local());
    EXPECT_EQ(seg0.size(), 25);

    auto seg3 = sv.segment_for_rank(3);
    EXPECT_TRUE(seg3.is_remote());  // We are rank 0, seg3 is rank 3
    EXPECT_EQ(seg3.size(), 25);
}

// =============================================================================
// Utility Methods
// =============================================================================

TEST(SegmentedViewTest, ForEachLocal) {
    distributed_vector<int> vec(10, 1, test_context{0, 1});
    auto sv = vec.segmented_view();

    sv.for_each_local([](int& x) { x *= 2; });

    auto local = vec.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 2);
    }
}

TEST(SegmentedViewTest, ForEachSegment) {
    distributed_vector<int> vec(10, 1, test_context{0, 1});
    auto sv = vec.segmented_view();

    int segment_count = 0;
    sv.for_each_segment([&segment_count](auto& seg) {
        ++segment_count;
        for (auto& elem : seg) {
            elem = 42;
        }
    });
    EXPECT_EQ(segment_count, 1);
    EXPECT_EQ(vec.local(0), 42);
}

// =============================================================================
// Const Correctness
// =============================================================================

TEST(SegmentedViewTest, ConstView) {
    distributed_vector<int> vec(10, 42, test_context{0, 1});
    const auto& cvec = vec;
    auto sv = cvec.segmented_view();

    auto seg = sv.local_segment();
    EXPECT_EQ(seg[0], 42);
    // Should not allow writes through const view (compile-time check)
    static_assert(std::is_const_v<std::remove_reference_t<decltype(seg[0])>>);
}

TEST(SegmentedViewTest, ConstIteration) {
    distributed_vector<int> vec(10, 7, test_context{0, 1});
    const auto& cvec = vec;
    auto sv = cvec.segmented_view();

    for (auto seg : sv) {
        for (const auto& elem : seg) {
            EXPECT_EQ(elem, 7);
        }
    }
}

// =============================================================================
// Type Trait Tests
// =============================================================================

TEST(SegmentedViewTest, TypeTrait) {
    using vec_type = distributed_vector<int>;
    using sv_type = segmented_view<vec_type>;
    static_assert(is_segmented_view<sv_type>::value);
    static_assert(!is_segmented_view<vec_type>::value);
}

}  // namespace dtl::test
