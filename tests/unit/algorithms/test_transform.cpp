// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_transform.cpp
/// @brief Unit tests for transform algorithm
/// @details Tests for Task 3.3: Modifying Algorithms

#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/views/local_view.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace dtl::test {

// =============================================================================
// Unary Transform Tests
// =============================================================================

TEST(TransformTest, UnaryDoubleValues) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(result, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(TransformTest, UnarySquare) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * x; });

    EXPECT_EQ(result, (std::vector<int>{1, 4, 9, 16, 25}));
}

TEST(TransformTest, UnaryNegate) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       std::negate<>{});

    EXPECT_EQ(result, (std::vector<int>{-1, -2, -3, -4, -5}));
}

TEST(TransformTest, InPlaceTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    dispatch_transform(seq{}, data.begin(), data.end(), data.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(TransformTest, EmptyRange) {
    std::vector<int> data;
    std::vector<int> result;

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 2; });

    EXPECT_TRUE(result.empty());
}

TEST(TransformTest, SingleElement) {
    std::vector<int> data = {42};
    std::vector<int> result(1);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(result[0], 84);
}

// =============================================================================
// Parallel Transform Tests
// =============================================================================

TEST(TransformTest, ParallelDoubleValues) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(par{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(result, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(TransformTest, ParallelInPlace) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    dispatch_transform(par{}, data.begin(), data.end(), data.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(TransformTest, LargeDataSetParallel) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 0);
    std::vector<int> result(10000);

    dispatch_transform(par{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 2; });

    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(result[i], static_cast<int>(i * 2));
    }
}

// =============================================================================
// Transform with Local View Tests
// =============================================================================

TEST(TransformTest, LocalViewTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);
    local_view<int> src_view(data.data(), data.size());
    local_view<int> dst_view(result.data(), result.size());

    dispatch_transform(seq{}, src_view.begin(), src_view.end(), dst_view.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(result, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(TransformTest, LocalViewInPlace) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    dispatch_transform(seq{}, view.begin(), view.end(), view.begin(),
                       [](int x) { return x * 2; });

    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

// =============================================================================
// Different Value Types
// =============================================================================

TEST(TransformTest, DoubleValues) {
    std::vector<double> data = {1.5, 2.5, 3.5};
    std::vector<double> result(3);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](double x) { return x * 2.0; });

    EXPECT_DOUBLE_EQ(result[0], 3.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 7.0);
}

TEST(TransformTest, FloatSqrt) {
    std::vector<float> data = {1.0f, 4.0f, 9.0f, 16.0f};
    std::vector<float> result(4);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](float x) { return std::sqrt(x); });

    EXPECT_FLOAT_EQ(result[0], 1.0f);
    EXPECT_FLOAT_EQ(result[1], 2.0f);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
}

TEST(TransformTest, StringTransform) {
    std::vector<std::string> data = {"hello", "world"};
    std::vector<size_t> result(2);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](const std::string& s) { return s.length(); });

    EXPECT_EQ(result[0], 5u);
    EXPECT_EQ(result[1], 5u);
}

TEST(TransformTest, IntToString) {
    std::vector<int> data = {1, 2, 3};
    std::vector<std::string> result(3);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return std::to_string(x); });

    EXPECT_EQ(result[0], "1");
    EXPECT_EQ(result[1], "2");
    EXPECT_EQ(result[2], "3");
}

// =============================================================================
// Binary Transform Tests
// =============================================================================

TEST(BinaryTransformTest, AddTwoVectors) {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30, 40, 50};
    std::vector<int> result(5);

    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});

    EXPECT_EQ(result, (std::vector<int>{11, 22, 33, 44, 55}));
}

TEST(BinaryTransformTest, MultiplyTwoVectors) {
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {2, 3, 4, 5, 6};
    std::vector<int> result(5);

    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<>{});

    EXPECT_EQ(result, (std::vector<int>{2, 6, 12, 20, 30}));
}

TEST(BinaryTransformTest, MaxOfTwoVectors) {
    std::vector<int> a = {1, 5, 3, 8, 2};
    std::vector<int> b = {3, 2, 7, 4, 9};
    std::vector<int> result(5);

    auto max_op = [](int x, int y) { return std::max(x, y); };
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), max_op);

    EXPECT_EQ(result, (std::vector<int>{3, 5, 7, 8, 9}));
}

// =============================================================================
// Complex Transform Operations
// =============================================================================

TEST(TransformTest, ChainedOperations) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    // First transform: square
    dispatch_transform(seq{}, data.begin(), data.end(), data.begin(),
                       [](int x) { return x * x; });

    // Second transform: negate
    dispatch_transform(seq{}, data.begin(), data.end(), data.begin(),
                       std::negate<>{});

    EXPECT_EQ(data, (std::vector<int>{-1, -4, -9, -16, -25}));
}

TEST(TransformTest, ConditionalTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    // Double even numbers, negate odd numbers
    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return (x % 2 == 0) ? x * 2 : -x; });

    EXPECT_EQ(result, (std::vector<int>{-1, 4, -3, 8, -5}));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(TransformTest, IdentityTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x; });

    EXPECT_EQ(result, data);
}

TEST(TransformTest, ConstantTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int) { return 42; });

    EXPECT_EQ(result, (std::vector<int>{42, 42, 42, 42, 42}));
}

TEST(TransformTest, ZeroTransform) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> result(5);

    dispatch_transform(seq{}, data.begin(), data.end(), result.begin(),
                       [](int x) { return x * 0; });

    EXPECT_EQ(result, (std::vector<int>{0, 0, 0, 0, 0}));
}

// =============================================================================
// Sequential vs Parallel Consistency
// =============================================================================

TEST(TransformTest, SeqAndParProduceSameResult) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);

    std::vector<int> seq_result(1000);
    std::vector<int> par_result(1000);

    auto square = [](int x) { return x * x; };

    dispatch_transform(seq{}, data.begin(), data.end(), seq_result.begin(), square);
    dispatch_transform(par{}, data.begin(), data.end(), par_result.begin(), square);

    EXPECT_EQ(seq_result, par_result);
}

// =============================================================================
// Stateful Function Object Tests
// =============================================================================

TEST(TransformTest, StatefulFunctionObject) {
    struct Counter {
        int count = 0;
        int operator()(int x) {
            return x + (count++);
        }
    };

    std::vector<int> data = {10, 20, 30, 40, 50};
    std::vector<int> result(5);

    Counter counter;
    std::transform(data.begin(), data.end(), result.begin(), std::ref(counter));

    EXPECT_EQ(result, (std::vector<int>{10, 21, 32, 43, 54}));
    EXPECT_EQ(counter.count, 5);
}

}  // namespace dtl::test
