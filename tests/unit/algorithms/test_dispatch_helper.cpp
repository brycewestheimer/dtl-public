// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_dispatch_helper.cpp
/// @brief Unit tests for dispatch_algorithm helper
/// @details Phase 27 Task 27.3: Verify that dispatch_algorithm correctly selects
///          seq/par/async implementations based on execution policy type.

#include <dtl/algorithms/dispatch_helper.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::test {

// =============================================================================
// Three-Way Dispatch
// =============================================================================

TEST(DispatchHelperTest, SeqPolicySelectsSeqImpl) {
    auto result = dispatch_algorithm<seq>(
        []() { return std::string("seq"); },
        []() { return std::string("par"); },
        []() { return std::string("async"); }
    );
    EXPECT_EQ(result, "seq");
}

TEST(DispatchHelperTest, ParPolicySelectsParImpl) {
    auto result = dispatch_algorithm<par>(
        []() { return std::string("seq"); },
        []() { return std::string("par"); },
        []() { return std::string("async"); }
    );
    EXPECT_EQ(result, "par");
}

TEST(DispatchHelperTest, AsyncPolicySelectsAsyncImpl) {
    auto result = dispatch_algorithm<async>(
        []() { return std::string("seq"); },
        []() { return std::string("par"); },
        []() { return std::string("async"); }
    );
    EXPECT_EQ(result, "async");
}

// =============================================================================
// Two-Way Dispatch
// =============================================================================

TEST(DispatchHelperTest, TwoWaySeqFallback) {
    int result = dispatch_algorithm<seq>(
        []() { return 1; },
        []() { return 2; }
    );
    EXPECT_EQ(result, 1);
}

TEST(DispatchHelperTest, TwoWayParSelected) {
    int result = dispatch_algorithm<par>(
        []() { return 1; },
        []() { return 2; }
    );
    EXPECT_EQ(result, 2);
}

TEST(DispatchHelperTest, TwoWayAsyncFallsBackToSeq) {
    // When there is no async impl, async policy falls back to seq
    int result = dispatch_algorithm<async>(
        []() { return 1; },
        []() { return 2; }
    );
    EXPECT_EQ(result, 1);
}

// =============================================================================
// Return Type Propagation
// =============================================================================

TEST(DispatchHelperTest, ReturnTypeIsPropagated) {
    auto result = dispatch_algorithm<seq>(
        []() -> double { return 3.14; },
        []() -> double { return 2.72; },
        []() -> double { return 1.41; }
    );
    EXPECT_DOUBLE_EQ(result, 3.14);
}

// =============================================================================
// Void Return
// =============================================================================

TEST(DispatchHelperTest, VoidReturnWorks) {
    int side_effect = 0;

    dispatch_algorithm<par>(
        [&]() { side_effect = 1; },
        [&]() { side_effect = 2; },
        [&]() { side_effect = 3; }
    );

    EXPECT_EQ(side_effect, 2);
}

}  // namespace dtl::test
