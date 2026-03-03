// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_backend_context_lifecycle.cpp
/// @brief Verify backend_context base class lifecycle methods
/// @details Tests that backend_context<T> transitions through states correctly
///          and that initialize/finalize provide default no-op behavior.

#include <dtl/backend/common/backend_context.hpp>
#include <dtl/backend/common/backend_traits.hpp>
#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Test Backend Tag
// =============================================================================

struct lifecycle_test_backend_tag {};

}  // namespace dtl::test

namespace dtl {

/// @brief Traits specialization for lifecycle test backend
template <>
struct backend_traits<test::lifecycle_test_backend_tag> {
    static constexpr bool supports_point_to_point = false;
    static constexpr bool supports_collectives = false;
    static constexpr bool supports_rma = false;
    static constexpr bool supports_gpu_aware = false;
    static constexpr bool supports_async = false;
    static constexpr bool supports_thread_multiple = false;
    static constexpr bool supports_rdma = false;
    static constexpr const char* name = "lifecycle_test";
    static constexpr backend_maturity maturity = backend_maturity::stub;
};

}  // namespace dtl

namespace dtl::test {

using test_context = dtl::backend_context<lifecycle_test_backend_tag>;

// =============================================================================
// State Transition Tests
// =============================================================================

TEST(BackendContextLifecycle, StartsUninitialized) {
    test_context ctx;
    EXPECT_EQ(ctx.state(), context_state::uninitialized);
    EXPECT_FALSE(ctx.is_active());
}

TEST(BackendContextLifecycle, InitializeTransitionsToActive) {
    test_context ctx;
    auto r = ctx.initialize();
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(ctx.state(), context_state::active);
    EXPECT_TRUE(ctx.is_active());
}

TEST(BackendContextLifecycle, FinalizeTransitionsToFinalized) {
    test_context ctx;
    ctx.initialize();
    auto r = ctx.finalize();
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(ctx.state(), context_state::finalized);
    EXPECT_FALSE(ctx.is_active());
}

TEST(BackendContextLifecycle, DoubleInitializeFails) {
    test_context ctx;
    ctx.initialize();
    auto r = ctx.initialize();
    EXPECT_FALSE(r.has_value());
}

TEST(BackendContextLifecycle, FinalizeWithoutInitializeFails) {
    test_context ctx;
    auto r = ctx.finalize();
    EXPECT_FALSE(r.has_value());
}

TEST(BackendContextLifecycle, DoubleFinalizeFailsSecondTime) {
    test_context ctx;
    ctx.initialize();
    ctx.finalize();
    auto r = ctx.finalize();
    EXPECT_FALSE(r.has_value());
}

TEST(BackendContextLifecycle, BackendNameMatchesTraits) {
    test_context ctx;
    EXPECT_STREQ(ctx.backend_name(), "lifecycle_test");
}

// =============================================================================
// Scoped Context Tests
// =============================================================================

TEST(BackendContextLifecycle, ScopedContextAutoInitializes) {
    {
        scoped_context<test_context> scoped;
        EXPECT_TRUE(scoped.get().is_active());
        EXPECT_STREQ(scoped->backend_name(), "lifecycle_test");
    }
    // Destructor should have finalized
    SUCCEED();
}

// =============================================================================
// make_context Helper Tests
// =============================================================================

TEST(BackendContextLifecycle, MakeContextReturnsActiveContext) {
    auto result = make_context<lifecycle_test_backend_tag>();
    EXPECT_TRUE(result.has_value());
    EXPECT_TRUE(result.value().is_active());
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(BackendContextLifecycle, MoveConstructTransfersOwnership) {
    test_context ctx;
    ctx.initialize();
    EXPECT_TRUE(ctx.is_active());

    test_context moved(std::move(ctx));
    EXPECT_TRUE(moved.is_active());
    EXPECT_EQ(ctx.state(), context_state::finalized);
}

TEST(BackendContextLifecycle, MoveAssignTransfersOwnership) {
    test_context ctx;
    ctx.initialize();

    test_context target;
    target = std::move(ctx);
    EXPECT_TRUE(target.is_active());
    EXPECT_EQ(ctx.state(), context_state::finalized);
}

}  // namespace dtl::test
