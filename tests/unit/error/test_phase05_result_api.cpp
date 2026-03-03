// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_phase05_result_api.cpp
/// @brief Phase 05 tests: result<T> API naming unification
/// @details Tests for T07 — transform_error() on both result<T> and result<void>,
///          and deprecated map_error() aliases.

#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

// Suppress deprecation warnings for testing the deprecated map_error()
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace dtl::test {

// =============================================================================
// result<T>::transform_error() tests
// =============================================================================

TEST(Phase05ResultApiTest, TransformErrorOnResultT_Error) {
    result<int> r(status{status_code::timeout, 5, "original"});
    auto transformed = r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "wrapped: " + s.message()};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
    EXPECT_EQ(transformed.error().rank(), 5);
    EXPECT_EQ(transformed.error().message(), "wrapped: original");
}

TEST(Phase05ResultApiTest, TransformErrorOnResultT_Success) {
    result<int> r(42);
    auto transformed = r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "should not be called"};
    });

    EXPECT_TRUE(transformed.has_value());
    EXPECT_EQ(transformed.value(), 42);
}

// =============================================================================
// result<void>::transform_error() tests (already existed)
// =============================================================================

TEST(Phase05ResultApiTest, TransformErrorOnResultVoid_Error) {
    result<void> r(status{status_code::timeout, 3, "void error"});
    auto transformed = r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "wrapped"};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
    EXPECT_EQ(transformed.error().rank(), 3);
}

TEST(Phase05ResultApiTest, TransformErrorOnResultVoid_Success) {
    result<void> r;
    auto transformed = r.transform_error([](status /*s*/) {
        return status{status_code::internal_error};
    });

    EXPECT_TRUE(transformed.has_value());
}

// =============================================================================
// Deprecated map_error() still works on result<T>
// =============================================================================

TEST(Phase05ResultApiTest, DeprecatedMapErrorOnResultT) {
    result<int> r(status{status_code::timeout, 1});
    auto transformed = r.map_error([](status s) {
        return status{status_code::internal_error, s.rank()};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
}

// =============================================================================
// Deprecated map_error() works on result<void>
// =============================================================================

TEST(Phase05ResultApiTest, DeprecatedMapErrorOnResultVoid) {
    result<void> r(status{status_code::timeout, 2});
    auto transformed = r.map_error([](status s) {
        return status{status_code::internal_error, s.rank()};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
}

// =============================================================================
// Generic code using transform_error() works for both specializations
// =============================================================================

template <typename T>
auto wrap_error(const result<T>& r) {
    return r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "wrapped"};
    });
}

TEST(Phase05ResultApiTest, GenericTransformError_ResultT) {
    result<int> r(status{status_code::timeout});
    auto wrapped = wrap_error(r);

    EXPECT_TRUE(wrapped.has_error());
    EXPECT_EQ(wrapped.error().code(), status_code::internal_error);
}

TEST(Phase05ResultApiTest, GenericTransformError_ResultVoid) {
    result<void> r(status{status_code::timeout});
    auto wrapped = wrap_error(r);

    EXPECT_TRUE(wrapped.has_error());
    EXPECT_EQ(wrapped.error().code(), status_code::internal_error);
}

}  // namespace dtl::test

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
