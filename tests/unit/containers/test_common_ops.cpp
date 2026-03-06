// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_common_ops.cpp
/// @brief Unit tests for common container operations
/// @details Phase 27 Task 27.2: Verify standalone_barrier, standalone_fence,
///          sync_container, rank_invariant_holds, and view construction helpers.

#include <dtl/containers/detail/common_ops.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/containers/distributed_array.hpp>
#include <dtl/containers/distributed_tensor.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

namespace {

struct test_context {
    rank_t my_rank = 0;
    rank_t num_ranks = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks; }
};

template <typename Container>
concept has_helper_local_view = requires(Container& container) {
    detail::make_local_view(container);
};

template <typename Container>
concept has_helper_const_local_view = requires(const Container& container) {
    detail::make_const_local_view(container);
};

template <typename Container>
concept has_helper_device_view = requires(Container& container) {
    detail::make_device_view(container);
};

}  // namespace

// =============================================================================
// Standalone Barrier / Fence
// =============================================================================

TEST(CommonOpsTest, StandaloneBarrierSucceeds) {
    auto r = detail::standalone_barrier();
    EXPECT_TRUE(r);
}

TEST(CommonOpsTest, StandaloneFenceSucceeds) {
    auto r = detail::standalone_fence();
    EXPECT_TRUE(r);
}

// =============================================================================
// Sync Container
// =============================================================================

TEST(CommonOpsTest, SyncContainerMarksClean) {
    distributed_vector<int> vec(100, 42);

    // Dirty the container
    vec.mark_local_modified();
    EXPECT_TRUE(vec.is_dirty());

    // Sync should succeed and mark clean
    auto r = detail::sync_container(vec);
    EXPECT_TRUE(r);
    EXPECT_TRUE(vec.is_clean());
}

TEST(CommonOpsTest, SyncContainerOnCleanIsNoOp) {
    distributed_vector<int> vec(100);
    EXPECT_TRUE(vec.is_clean());

    auto r = detail::sync_container(vec);
    EXPECT_TRUE(r);
    EXPECT_TRUE(vec.is_clean());
}

// =============================================================================
// Rank Invariant
// =============================================================================

TEST(CommonOpsTest, RankInvariantHoldsForSingleRank) {
    test_context ctx{0, 1};
    distributed_vector<int> vec(100, ctx);
    EXPECT_TRUE(detail::rank_invariant_holds(vec));
}

TEST(CommonOpsTest, RankInvariantHoldsForMultiRank) {
    test_context ctx{2, 4};
    distributed_vector<int> vec(100, ctx);
    EXPECT_TRUE(detail::rank_invariant_holds(vec));
}

// =============================================================================
// View Construction Helpers - Vector
// =============================================================================

TEST(CommonOpsTest, MakeLocalViewForVector) {
    distributed_vector<int> vec(100, 7);
    auto view = detail::make_local_view(vec);

    EXPECT_EQ(view.size(), vec.local_size());
    EXPECT_EQ(view.data(), vec.local_data());
}

TEST(CommonOpsTest, MakeConstLocalViewForVector) {
    const distributed_vector<int> vec(100, 7);
    auto view = detail::make_const_local_view(vec);

    EXPECT_EQ(view.size(), vec.local_size());
    EXPECT_EQ(view.data(), vec.local_data());
}

TEST(CommonOpsTest, MakeLocalViewMatchesContainerLocalView) {
    distributed_vector<int> vec(100, 42);
    auto helper_view = detail::make_local_view(vec);
    auto direct_view = vec.local_view();

    EXPECT_EQ(helper_view.size(), direct_view.size());
    EXPECT_EQ(helper_view.data(), direct_view.data());
}

// =============================================================================
// View Construction Helpers - Array
// =============================================================================

TEST(CommonOpsTest, MakeLocalViewForArray) {
    distributed_array<int, 50> arr;
    auto view = detail::make_local_view(arr);

    EXPECT_EQ(view.size(), arr.local_size());
    EXPECT_EQ(view.data(), arr.local_data());
}

#if DTL_ENABLE_CUDA

TEST(CommonOpsTest, DeviceOnlyHelpersExposeDeviceViewOnly) {
    using vec_t = distributed_vector<int, device_only<0>>;

    static_assert(!has_helper_local_view<vec_t>);
    static_assert(!has_helper_const_local_view<vec_t>);
    static_assert(has_helper_device_view<vec_t>);
}

TEST(CommonOpsTest, UnifiedMemoryHelpersExposeBothViewKinds) {
    using vec_t = distributed_vector<int, unified_memory>;

    static_assert(has_helper_local_view<vec_t>);
    static_assert(has_helper_const_local_view<vec_t>);
    static_assert(has_helper_device_view<vec_t>);
}

#endif

}  // namespace dtl::test
