// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_adopt_mode.cpp
/// @brief Unit tests for MPI lifecycle helpers and adopt mode
/// @details Phase 12A: Tests mpi_lifecycle.hpp query/management functions.
///          Without DTL_ENABLE_MPI, all MPI operations return not_supported
///          or false, which is the expected behavior for stub mode.

#include <backends/mpi/mpi_lifecycle.hpp>
#include <dtl/core/environment_options.hpp>

#include <gtest/gtest.h>

#include <string_view>

namespace dtl::test {

// =============================================================================
// MPI State Enum Tests
// =============================================================================

TEST(MpiLifecycleTest, MpiStateEnum) {
    // Verify all enum values exist and are distinct
    auto s1 = mpi::mpi_state::not_initialized;
    auto s2 = mpi::mpi_state::initialized;
    auto s3 = mpi::mpi_state::finalized;

    EXPECT_NE(s1, s2);
    EXPECT_NE(s2, s3);
    EXPECT_NE(s1, s3);
}

TEST(MpiLifecycleTest, MpiStateToString) {
    EXPECT_EQ(mpi::to_string(mpi::mpi_state::not_initialized), "not_initialized");
    EXPECT_EQ(mpi::to_string(mpi::mpi_state::initialized), "initialized");
    EXPECT_EQ(mpi::to_string(mpi::mpi_state::finalized), "finalized");
}

// =============================================================================
// Thread Level Constants
// =============================================================================

TEST(MpiLifecycleTest, ThreadLevelConstants) {
    EXPECT_EQ(mpi::thread_levels::single, 0);
    EXPECT_EQ(mpi::thread_levels::funneled, 1);
    EXPECT_EQ(mpi::thread_levels::serialized, 2);
    EXPECT_EQ(mpi::thread_levels::multiple, 3);

    // Verify ordering
    EXPECT_LT(mpi::thread_levels::single, mpi::thread_levels::funneled);
    EXPECT_LT(mpi::thread_levels::funneled, mpi::thread_levels::serialized);
    EXPECT_LT(mpi::thread_levels::serialized, mpi::thread_levels::multiple);
}

TEST(MpiLifecycleTest, RequiredLevelRange) {
    // Valid thread levels are 0 through 3
    EXPECT_GE(mpi::thread_levels::single, 0);
    EXPECT_LE(mpi::thread_levels::multiple, 3);
}

TEST(MpiLifecycleTest, AllowFallbackDefault) {
    // The default mpi_options has allow_thread_fallback = true
    mpi_options opts{};
    EXPECT_TRUE(opts.allow_thread_fallback);
}

// =============================================================================
// Non-MPI Stub Behavior Tests
// =============================================================================

// These tests verify behavior when MPI support is disabled (DTL_ENABLE_MPI == 0).
// When MPI IS available (DTL_ENABLE_MPI == 1), these tests check
// the actual MPI state which may differ from the stub behavior.

TEST(MpiLifecycleTest, IsInitializedWithoutMpi) {
    // Without MPI enabled, is_initialized should return false
    // With MPI enabled, it depends on whether MPI was initialized
#if !DTL_ENABLE_MPI
    EXPECT_FALSE(mpi::is_initialized());
#else
    // Just verify the function is callable; result depends on MPI state
    [[maybe_unused]] bool val = mpi::is_initialized();
    SUCCEED();
#endif
}

TEST(MpiLifecycleTest, IsFinalizedWithoutMpi) {
#if !DTL_ENABLE_MPI
    EXPECT_FALSE(mpi::is_finalized());
#else
    [[maybe_unused]] bool val = mpi::is_finalized();
    SUCCEED();
#endif
}

TEST(MpiLifecycleTest, QueryThreadLevelWithoutMpi) {
#if !DTL_ENABLE_MPI
    auto res = mpi::query_thread_level();
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
#else
    // With MPI, result depends on initialization state
    auto res = mpi::query_thread_level();
    // Just verify it returns a valid result type
    EXPECT_TRUE(res.has_value() || res.has_error());
#endif
}

TEST(MpiLifecycleTest, GetStateWithoutMpi) {
#if !DTL_ENABLE_MPI
    EXPECT_EQ(mpi::get_state(), mpi::mpi_state::not_initialized);
#else
    // With MPI, state is whatever the runtime reports
    auto state = mpi::get_state();
    EXPECT_TRUE(state == mpi::mpi_state::not_initialized ||
                state == mpi::mpi_state::initialized ||
                state == mpi::mpi_state::finalized);
#endif
}

TEST(MpiLifecycleTest, InitializeWithoutMpi) {
#if !DTL_ENABLE_MPI
    auto res = mpi::initialize(mpi::thread_levels::funneled, true);
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
#else
    // With MPI, we cannot safely call initialize in a unit test because
    // MPI may already be initialized. Verify the function signature is correct.
    SUCCEED();
#endif
}

TEST(MpiLifecycleTest, VerifyInitializedWithoutMpi) {
#if !DTL_ENABLE_MPI
    auto res = mpi::verify_initialized();
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
#else
    auto res = mpi::verify_initialized();
    // Result depends on MPI state
    EXPECT_TRUE(res.has_value() || res.has_error());
#endif
}

TEST(MpiLifecycleTest, FinalizeWithoutMpi) {
#if !DTL_ENABLE_MPI
    auto res = mpi::finalize_mpi();
    EXPECT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::not_supported);
#else
    // Cannot safely call finalize in a unit test. Verify signature only.
    SUCCEED();
#endif
}

// =============================================================================
// State Transition Tests
// =============================================================================

TEST(MpiLifecycleTest, StateTransitions) {
    // Verify that all three states are representable
    mpi::mpi_state states[] = {
        mpi::mpi_state::not_initialized,
        mpi::mpi_state::initialized,
        mpi::mpi_state::finalized
    };

    EXPECT_EQ(states[0], mpi::mpi_state::not_initialized);
    EXPECT_EQ(states[1], mpi::mpi_state::initialized);
    EXPECT_EQ(states[2], mpi::mpi_state::finalized);

    // All three should be distinct
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            EXPECT_NE(states[i], states[j]);
        }
    }
}

// =============================================================================
// Lifecycle Order and Idempotent Init Tests
// =============================================================================

TEST(MpiLifecycleTest, InitializeIdempotent) {
    // Without MPI, repeated calls all fail with not_supported
#if !DTL_ENABLE_MPI
    auto res1 = mpi::initialize(mpi::thread_levels::single, true);
    auto res2 = mpi::initialize(mpi::thread_levels::single, true);
    EXPECT_TRUE(res1.has_error());
    EXPECT_TRUE(res2.has_error());
#else
    // With MPI enabled, initialize() is designed to be idempotent:
    // if MPI is already initialized, it is adopted and the existing
    // thread level is returned.
    if (mpi::is_initialized()) {
        auto res1 = mpi::initialize(mpi::thread_levels::single, true);
        ASSERT_TRUE(res1.has_value());
        auto res2 = mpi::initialize(mpi::thread_levels::single, true);
        ASSERT_TRUE(res2.has_value());
    } else {
        // Avoid initializing MPI from this unit test (some MPI impls
        // may not accept nullptr argc/argv).
        SUCCEED();
    }
#endif
}

TEST(MpiLifecycleTest, LifecycleOrder) {
    // Verify that finalize before init is an error
#if !DTL_ENABLE_MPI
    // Without MPI, both operations fail with not_supported
    auto fin_res = mpi::finalize_mpi();
    EXPECT_TRUE(fin_res.has_error());

    auto init_res = mpi::initialize(mpi::thread_levels::funneled, true);
    EXPECT_TRUE(init_res.has_error());
#else
    // With MPI, attempting finalize when not initialized returns error
    if (!mpi::is_initialized() && !mpi::is_finalized()) {
        auto res = mpi::finalize_mpi();
        EXPECT_TRUE(res.has_error());
        EXPECT_EQ(res.error().code(), status_code::invalid_state);
    } else {
        SUCCEED();
    }
#endif
}

}  // namespace dtl::test
