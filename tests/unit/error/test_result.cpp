// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_result.cpp
/// @brief Unit tests for dtl/error/result.hpp
/// @details Tests result type including result<T> and result<void>.

#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::test {

// =============================================================================
// Result<T> Construction Tests
// =============================================================================

TEST(ResultTest, DefaultConstruction) {
    result<int> r;
    EXPECT_TRUE(r.has_value());
    EXPECT_FALSE(r.has_error());
    EXPECT_EQ(r.value(), 0);  // Default-constructed int
}

TEST(ResultTest, ValueConstruction) {
    result<int> r(42);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultTest, ValueConstructionMove) {
    result<std::string> r(std::string("hello"));
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), "hello");
}

TEST(ResultTest, ErrorStatusConstruction) {
    result<int> r(status{status_code::allocation_failed});
    EXPECT_TRUE(r.has_error());
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code(), status_code::allocation_failed);
}

TEST(ResultTest, ErrorCodeConstruction) {
    result<int> r(status_code::timeout);
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::timeout);
}

// =============================================================================
// Result<T> Observer Tests
// =============================================================================

TEST(ResultTest, BoolConversion) {
    result<int> success(42);
    result<int> error(status_code::timeout);

    EXPECT_TRUE(static_cast<bool>(success));
    EXPECT_FALSE(static_cast<bool>(error));
}

TEST(ResultTest, ValueAccess) {
    result<int> r(42);

    EXPECT_EQ(r.value(), 42);
    EXPECT_EQ(*r, 42);
}

TEST(ResultTest, ValueMutation) {
    result<int> r(10);
    r.value() = 20;
    EXPECT_EQ(r.value(), 20);
}

TEST(ResultTest, ValueOr) {
    result<int> success(42);
    result<int> error(status_code::timeout);

    EXPECT_EQ(success.value_or(0), 42);
    EXPECT_EQ(error.value_or(0), 0);
}

TEST(ResultTest, ArrowOperator) {
    result<std::string> r("hello");
    EXPECT_EQ(r->size(), 5);
}

// =============================================================================
// Result<T> Monadic Operations Tests
// =============================================================================

TEST(ResultTest, MapSuccess) {
    result<int> r(21);
    auto doubled = r.map([](int x) { return x * 2; });

    EXPECT_TRUE(doubled.has_value());
    EXPECT_EQ(doubled.value(), 42);
}

TEST(ResultTest, MapError) {
    result<int> r(status_code::timeout);
    auto doubled = r.map([](int x) { return x * 2; });

    EXPECT_TRUE(doubled.has_error());
    EXPECT_EQ(doubled.error().code(), status_code::timeout);
}

TEST(ResultTest, MapTypeChange) {
    result<int> r(42);
    auto str = r.map([](int x) { return std::to_string(x); });

    EXPECT_TRUE(str.has_value());
    EXPECT_EQ(str.value(), "42");
}

TEST(ResultTest, AndThenSuccess) {
    result<int> r(10);
    auto chained = r.and_then([](int x) -> result<int> {
        if (x > 0) {
            return x * 2;
        }
        return status_code::invalid_argument;
    });

    EXPECT_TRUE(chained.has_value());
    EXPECT_EQ(chained.value(), 20);
}

TEST(ResultTest, AndThenFailure) {
    result<int> r(10);
    auto chained = r.and_then([](int /*x*/) -> result<int> {
        return status_code::not_supported;
    });

    EXPECT_TRUE(chained.has_error());
    EXPECT_EQ(chained.error().code(), status_code::not_supported);
}

TEST(ResultTest, AndThenPropagatesError) {
    result<int> r(status_code::timeout);
    bool called = false;
    auto chained = r.and_then([&](int x) -> result<int> {
        called = true;
        return x * 2;
    });

    EXPECT_FALSE(called);
    EXPECT_TRUE(chained.has_error());
    EXPECT_EQ(chained.error().code(), status_code::timeout);
}

TEST(ResultTest, MapError_) {
    result<int> r(status{status_code::timeout, 5});
    auto transformed = r.transform_error([](status s) {
        return status{s.code(), s.rank(), "transformed: " + s.message()};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().rank(), 5);
}

TEST(ResultTest, OrElse) {
    result<int> error(status_code::timeout);
    int value = error.or_else([](const status& /*s*/) { return -1; });

    EXPECT_EQ(value, -1);
}

TEST(ResultTest, OrElseNotCalledOnSuccess) {
    result<int> success(42);
    bool called = false;
    int value = success.or_else([&](const status& /*s*/) {
        called = true;
        return -1;
    });

    EXPECT_FALSE(called);
    EXPECT_EQ(value, 42);
}

// =============================================================================
// Result<void> Tests
// =============================================================================

TEST(ResultVoidTest, DefaultConstruction) {
    result<void> r;
    EXPECT_TRUE(r.has_value());
    EXPECT_FALSE(r.has_error());
}

TEST(ResultVoidTest, ErrorConstruction) {
    result<void> r(status{status_code::send_failed});
    EXPECT_FALSE(r.has_value());
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::send_failed);
}

TEST(ResultVoidTest, ErrorCodeConstruction) {
    result<void> r(status_code::barrier_failed);
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::barrier_failed);
}

TEST(ResultVoidTest, BoolConversion) {
    result<void> success;
    result<void> error(status_code::timeout);

    EXPECT_TRUE(static_cast<bool>(success));
    EXPECT_FALSE(static_cast<bool>(error));
}

TEST(ResultVoidTest, MapSuccess) {
    result<void> r;
    auto mapped = r.map([]() { return 42; });

    EXPECT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped.value(), 42);
}

TEST(ResultVoidTest, MapError) {
    result<void> r(status_code::timeout);
    auto mapped = r.map([]() { return 42; });

    EXPECT_TRUE(mapped.has_error());
    EXPECT_EQ(mapped.error().code(), status_code::timeout);
}

TEST(ResultVoidTest, AndThenSuccess) {
    result<void> r;
    auto chained = r.and_then([]() -> result<int> {
        return 42;
    });

    EXPECT_TRUE(chained.has_value());
    EXPECT_EQ(chained.value(), 42);
}

TEST(ResultVoidTest, AndThenError) {
    result<void> r(status_code::timeout);
    bool called = false;
    auto chained = r.and_then([&]() -> result<int> {
        called = true;
        return 42;
    });

    EXPECT_FALSE(called);
    EXPECT_TRUE(chained.has_error());
}

TEST(ResultVoidTest, TransformErrorSuccess) {
    result<void> r;
    auto transformed = r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "wrapped"};
    });

    EXPECT_TRUE(transformed.has_value());
}

TEST(ResultVoidTest, TransformErrorError) {
    result<void> r(status{status_code::timeout, 3});
    auto transformed = r.transform_error([](status s) {
        return status{status_code::internal_error, s.rank(), "wrapped"};
    });

    EXPECT_TRUE(transformed.has_error());
    EXPECT_EQ(transformed.error().code(), status_code::internal_error);
    EXPECT_EQ(transformed.error().rank(), 3);
}

TEST(ResultVoidTest, OrElseError) {
    result<void> r(status_code::timeout);
    bool called = false;
    r.or_else([&](const status& s) {
        called = true;
        EXPECT_EQ(s.code(), status_code::timeout);
    });

    EXPECT_TRUE(called);
}

TEST(ResultVoidTest, OrElseSuccess) {
    result<void> r;
    bool called = false;
    r.or_else([&](const status& /*s*/) {
        called = true;
    });

    EXPECT_FALSE(called);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(ResultFactoryTest, MakeResult) {
    auto r = make_result(42);
    static_assert(std::is_same_v<decltype(r), result<int>>);
    EXPECT_TRUE(r.has_value());
    EXPECT_EQ(r.value(), 42);
}

TEST(ResultFactoryTest, MakeOkResult) {
    auto r = make_ok_result();
    static_assert(std::is_same_v<decltype(r), result<void>>);
    EXPECT_TRUE(r.has_value());
}

TEST(ResultFactoryTest, MakeErrorResult) {
    auto r = make_error_result<int>(status{status_code::timeout});
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::timeout);
}

TEST(ResultFactoryTest, MakeErrorResultFromCode) {
    auto r = make_error_result<std::string>(status_code::not_implemented);
    EXPECT_TRUE(r.has_error());
    EXPECT_EQ(r.error().code(), status_code::not_implemented);
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(ResultTest, MoveValue) {
    result<std::string> r1("hello world");
    result<std::string> r2 = std::move(r1).map([](std::string s) { return s + "!"; });

    EXPECT_TRUE(r2.has_value());
    EXPECT_EQ(r2.value(), "hello world!");
}

TEST(ResultTest, MoveOutValue) {
    result<std::string> r("hello");
    std::string s = std::move(r).value();
    EXPECT_EQ(s, "hello");
}

}  // namespace dtl::test
