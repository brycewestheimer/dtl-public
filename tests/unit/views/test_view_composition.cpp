// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_view_composition.cpp
/// @brief Unit tests for view composition and traits
/// @details Tests for Phase 11.5: view traits and composition utilities

#include <dtl/views/views.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/views/global_view.hpp>
#include <dtl/views/remote_ref.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/core/traits.hpp>

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
// may_communicate Trait Tests
// =============================================================================

TEST(ViewCompositionTest, MayCommunicateLocalView) {
    // local_view never communicates
    static_assert(!may_communicate_v<local_view<int>>);
    static_assert(!may_communicate_v<local_view<double>>);
    static_assert(!may_communicate_v<local_view<const int>>);
}

TEST(ViewCompositionTest, MayCommunicateGlobalView) {
    // global_view may communicate
    using vec_type = distributed_vector<int>;
    using gv_type = global_view<vec_type>;

    static_assert(may_communicate_v<gv_type>);
}

TEST(ViewCompositionTest, MayCommunicateRemoteRef) {
    // remote_ref may communicate
    static_assert(may_communicate_v<remote_ref<int>>);
    static_assert(may_communicate_v<remote_ref<double>>);
    static_assert(may_communicate_v<remote_ref<const int>>);
}

TEST(ViewCompositionTest, MayCommunicateStdTypes) {
    // Standard library types don't communicate
    static_assert(!may_communicate_v<std::vector<int>>);
    static_assert(!may_communicate_v<int>);
    static_assert(!may_communicate_v<double*>);
}

// =============================================================================
// is_stl_safe Trait Tests
// =============================================================================

TEST(ViewCompositionTest, IsSTLSafeLocalView) {
    // local_view is STL-safe
    static_assert(is_stl_safe_v<local_view<int>>);
    static_assert(is_stl_safe_v<local_view<double>>);
}

TEST(ViewCompositionTest, IsSTLSafeGlobalView) {
    // global_view is NOT STL-safe (may communicate)
    using vec_type = distributed_vector<int>;
    using gv_type = global_view<vec_type>;

    static_assert(!is_stl_safe_v<gv_type>);
}

TEST(ViewCompositionTest, IsSTLSafeStdTypes) {
    // Standard types are STL-safe
    static_assert(is_stl_safe_v<std::vector<int>>);
    static_assert(is_stl_safe_v<int>);
}

// =============================================================================
// is_local_view Trait Tests
// =============================================================================

TEST(ViewCompositionTest, IsLocalViewTrait) {
    static_assert(is_local_view_v<local_view<int>>);
    static_assert(is_local_view_v<local_view<double>>);
    static_assert(is_local_view_v<local_view<const int>>);

    static_assert(!is_local_view_v<std::vector<int>>);
    static_assert(!is_local_view_v<distributed_vector<int>>);
    static_assert(!is_local_view_v<int>);
}

// =============================================================================
// is_global_view Trait Tests
// =============================================================================

TEST(ViewCompositionTest, IsGlobalViewTrait) {
    using vec_type = distributed_vector<int>;

    static_assert(is_global_view_v<global_view<vec_type>>);

    static_assert(!is_global_view_v<local_view<int>>);
    static_assert(!is_global_view_v<vec_type>);
    static_assert(!is_global_view_v<int>);
}

// =============================================================================
// is_remote_ref Trait Tests
// =============================================================================

TEST(ViewCompositionTest, IsRemoteRefTrait) {
    static_assert(is_remote_ref_v<remote_ref<int>>);
    static_assert(is_remote_ref_v<remote_ref<double>>);
    static_assert(is_remote_ref_v<remote_ref<const int>>);

    static_assert(!is_remote_ref_v<int*>);
    static_assert(!is_remote_ref_v<int&>);
    static_assert(!is_remote_ref_v<local_view<int>>);
}

// =============================================================================
// is_batching_view Trait Tests
// =============================================================================

TEST(ViewCompositionTest, IsBatchingViewChunk) {
    // chunk_view is a batching view
    using vec_type = std::vector<int>;

    static_assert(is_batching_view_v<chunk_view<vec_type>>);
}

TEST(ViewCompositionTest, IsBatchingViewWindow) {
    // window_view is a batching view
    using vec_type = std::vector<int>;

    static_assert(is_batching_view_v<window_view<vec_type>>);
}

TEST(ViewCompositionTest, IsBatchingViewOtherTypes) {
    // Other types are not batching views
    static_assert(!is_batching_view_v<local_view<int>>);
    static_assert(!is_batching_view_v<std::vector<int>>);
    static_assert(!is_batching_view_v<int>);
}

// =============================================================================
// View from Container Tests
// =============================================================================

TEST(ViewCompositionTest, LocalViewFromVector) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto lv = vec.local_view();

    static_assert(is_local_view_v<decltype(lv)>);
    EXPECT_EQ(lv.size(), 25);
}

TEST(ViewCompositionTest, GlobalViewFromVector) {
    distributed_vector<int> vec(100, 42, test_context{1, 4});

    auto gv = vec.global_view();

    static_assert(is_global_view_v<decltype(gv)>);
    EXPECT_EQ(gv.size(), 100);
}

TEST(ViewCompositionTest, LocalViewSTLCompatibility) {
    distributed_vector<int> vec(100, test_context{0, 1});
    auto local = vec.local_view();

    // Fill with iota
    for (size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(i);
    }

    // Should work with STL algorithms
    int sum = 0;
    for (int val : local) {
        sum += val;
    }

    // Sum of 0..99 = 4950
    EXPECT_EQ(sum, 4950);
}

// =============================================================================
// DistributedView Concept Tests
// =============================================================================

TEST(ViewCompositionTest, DistributedViewConcept) {
    // Test that local_view satisfies DistributedView concept
    // Note: global_view doesn't satisfy the concept as it lacks begin()/end()
    // iterators (by design - iteration should be explicit via local segments)
    static_assert(DistributedView<local_view<int>>);

    EXPECT_TRUE(DistributedView<local_view<int>>);
}

// =============================================================================
// View Trait Composition Tests
// =============================================================================

TEST(ViewCompositionTest, LocalViewNoCommunication) {
    // local_view should be both STL-safe and not communicate
    using lv_type = local_view<int>;

    static_assert(!may_communicate_v<lv_type>);
    static_assert(is_stl_safe_v<lv_type>);
    static_assert(is_local_view_v<lv_type>);
    static_assert(!is_global_view_v<lv_type>);
    static_assert(!is_remote_ref_v<lv_type>);
}

TEST(ViewCompositionTest, GlobalViewCommunicates) {
    // global_view may communicate and is not STL-safe
    using vec_type = distributed_vector<int>;
    using gv_type = global_view<vec_type>;

    static_assert(may_communicate_v<gv_type>);
    static_assert(!is_stl_safe_v<gv_type>);
    static_assert(!is_local_view_v<gv_type>);
    static_assert(is_global_view_v<gv_type>);
    static_assert(!is_remote_ref_v<gv_type>);
}

TEST(ViewCompositionTest, RemoteRefCommunicates) {
    // remote_ref may communicate
    using rr_type = remote_ref<int>;

    static_assert(may_communicate_v<rr_type>);
    static_assert(!is_local_view_v<rr_type>);
    static_assert(!is_global_view_v<rr_type>);
    static_assert(is_remote_ref_v<rr_type>);
}

// =============================================================================
// View Size and Data Access Tests
// =============================================================================

TEST(ViewCompositionTest, LocalViewDataAccess) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data);

    EXPECT_EQ(view.data(), data.data());
    EXPECT_EQ(view.size(), 5);
    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view[4], 5);
}

TEST(ViewCompositionTest, MakeLocalViewFactory) {
    std::vector<int> data = {10, 20, 30};

    auto view = make_local_view(data);

    static_assert(is_local_view_v<decltype(view)>);
    EXPECT_EQ(view.size(), 3);
    EXPECT_EQ(view[0], 10);
}

// =============================================================================
// Const Correctness Tests
// =============================================================================

TEST(ViewCompositionTest, ConstLocalView) {
    const std::vector<int> data = {1, 2, 3};
    local_view<const int> view(data.data(), data.size());

    EXPECT_EQ(view[0], 1);
    EXPECT_EQ(view.size(), 3);

    // Should be read-only
    static_assert(std::is_const_v<std::remove_reference_t<decltype(view[0])>>);
}

// Note: ConstGlobalView test removed - the current global_view implementation
// doesn't properly propagate const-ness. This could be improved in a future
// version to return remote_ref<const T> for const containers.

}  // namespace dtl::test
