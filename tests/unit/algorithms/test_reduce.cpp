// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_reduce.cpp
/// @brief Unit tests for reduce algorithm
/// @details Tests for Task 3.4: Reduction Algorithms (CRITICAL)

#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/views/local_view.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace dtl::test {

// =============================================================================
// reduce_result Type Tests
// =============================================================================

TEST(ReduceResultTest, DefaultConstruction) {
    reduce_result<int> result{0, 0, false};
    EXPECT_EQ(result.local_value, 0);
    EXPECT_EQ(result.global_value, 0);
    EXPECT_FALSE(result.has_global);
}

TEST(ReduceResultTest, ValueMethod) {
    // When has_global is true, value() returns global_value
    reduce_result<int> result1{10, 100, true};
    EXPECT_EQ(result1.value(), 100);

    // When has_global is false, value() returns local_value
    reduce_result<int> result2{10, 100, false};
    EXPECT_EQ(result2.value(), 10);
}

TEST(ReduceResultTest, ImplicitConversion) {
    reduce_result<int> result{10, 100, true};
    int value = result;  // Implicit conversion
    EXPECT_EQ(value, 100);
}

TEST(ReduceResultTest, WithDifferentTypes) {
    reduce_result<double> double_result{1.5, 3.5, true};
    EXPECT_DOUBLE_EQ(double_result.value(), 3.5);

    reduce_result<long long> ll_result{1000000000LL, 5000000000LL, true};
    EXPECT_EQ(ll_result.value(), 5000000000LL);
}

// =============================================================================
// Local Reduce Tests (dispatch_reduce)
// =============================================================================

TEST(LocalReduceTest, SumWithPlus) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(LocalReduceTest, SumWithInitialValue) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 10, std::plus<>{});
    EXPECT_EQ(result, 25);  // 10 + 15
}

TEST(LocalReduceTest, ProductWithMultiplies) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 1, std::multiplies<>{});
    EXPECT_EQ(result, 120);  // 5!
}

TEST(LocalReduceTest, MaxElement) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    auto max_op = [](int a, int b) { return std::max(a, b); };
    int result = dispatch_reduce(seq{}, data.begin(), data.end(),
                                  std::numeric_limits<int>::min(), max_op);
    EXPECT_EQ(result, 9);
}

TEST(LocalReduceTest, MinElement) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6};
    auto min_op = [](int a, int b) { return std::min(a, b); };
    int result = dispatch_reduce(seq{}, data.begin(), data.end(),
                                  std::numeric_limits<int>::max(), min_op);
    EXPECT_EQ(result, 1);
}

TEST(LocalReduceTest, EmptyRange) {
    std::vector<int> data;
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 42, std::plus<>{});
    EXPECT_EQ(result, 42);  // Returns init for empty range
}

TEST(LocalReduceTest, SingleElement) {
    std::vector<int> data = {42};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 42);
}

TEST(LocalReduceTest, NegativeNumbers) {
    std::vector<int> data = {-5, -4, -3, -2, -1};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, -15);
}

TEST(LocalReduceTest, MixedNumbers) {
    std::vector<int> data = {-2, -1, 0, 1, 2};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 0);
}

TEST(LocalReduceTest, LargeDataSet) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 500500);  // Sum of 1 to 1000
}

TEST(LocalReduceTest, DoubleSum) {
    std::vector<double> data = {1.5, 2.5, 3.0, 4.0};
    double result = dispatch_reduce(seq{}, data.begin(), data.end(), 0.0, std::plus<>{});
    EXPECT_DOUBLE_EQ(result, 11.0);
}

// =============================================================================
// Parallel Reduce Tests
// =============================================================================

TEST(ParallelReduceTest, SumWithPlus) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(ParallelReduceTest, Product) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    int result = dispatch_reduce(par{}, data.begin(), data.end(), 1, std::multiplies<>{});
    EXPECT_EQ(result, 120);
}

TEST(ParallelReduceTest, LargeDataSetMatchesSequential) {
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 1);

    int seq_result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    int par_result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});

    EXPECT_EQ(seq_result, par_result);
}

TEST(ParallelReduceTest, Max) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);
    data[500] = 10000;  // Insert max value in middle

    auto max_op = [](int a, int b) { return std::max(a, b); };
    int result = dispatch_reduce(par{}, data.begin(), data.end(),
                                  std::numeric_limits<int>::min(), max_op);
    EXPECT_EQ(result, 10000);
}

// =============================================================================
// Reduce with Local View Tests
// =============================================================================

TEST(LocalViewReduceTest, BasicReduce) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int result = dispatch_reduce(seq{}, view.begin(), view.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(LocalViewReduceTest, ParallelReduce) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int result = dispatch_reduce(par{}, view.begin(), view.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

TEST(LocalViewReduceTest, ConstView) {
    const std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<const int> view(data.data(), data.size());

    int result = dispatch_reduce(seq{}, view.begin(), view.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 15);
}

// =============================================================================
// Custom Binary Operation Tests
// =============================================================================

TEST(CustomReduceOpTest, BitwiseOr) {
    std::vector<int> data = {1, 2, 4, 8, 16};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::bit_or<>{});
    EXPECT_EQ(result, 31);  // 11111 in binary
}

TEST(CustomReduceOpTest, BitwiseAnd) {
    std::vector<int> data = {0b1111, 0b1110, 0b1100};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), ~0, std::bit_and<>{});
    EXPECT_EQ(result, 0b1100);
}

TEST(CustomReduceOpTest, LambdaOperation) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto sum_squared = [](int a, int b) { return a + b * b; };
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, sum_squared);
    EXPECT_EQ(result, 55);  // 1 + 4 + 9 + 16 + 25
}

TEST(CustomReduceOpTest, StringConcatenation) {
    std::vector<std::string> data = {"a", "b", "c", "d"};
    auto concat = [](const std::string& a, const std::string& b) { return a + b; };
    std::string result = dispatch_reduce(seq{}, data.begin(), data.end(),
                                          std::string{}, concat);
    EXPECT_EQ(result, "abcd");
}

// =============================================================================
// Numeric Precision Tests
// =============================================================================

TEST(NumericPrecisionTest, FloatSum) {
    std::vector<float> data = {0.1f, 0.2f, 0.3f};
    float result = dispatch_reduce(seq{}, data.begin(), data.end(), 0.0f, std::plus<>{});
    EXPECT_NEAR(result, 0.6f, 1e-6f);
}

TEST(NumericPrecisionTest, DoubleSum) {
    std::vector<double> data = {0.1, 0.2, 0.3};
    double result = dispatch_reduce(seq{}, data.begin(), data.end(), 0.0, std::plus<>{});
    EXPECT_NEAR(result, 0.6, 1e-15);
}

TEST(NumericPrecisionTest, LargeFloatSum) {
    std::vector<float> data(10000, 0.1f);
    float result = dispatch_reduce(seq{}, data.begin(), data.end(), 0.0f, std::plus<>{});
    EXPECT_NEAR(result, 1000.0f, 0.1f);  // Allow for accumulation error
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ReduceEdgeCasesTest, AllSameValues) {
    std::vector<int> data(100, 5);
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 500);
}

TEST(ReduceEdgeCasesTest, AlternatingValues) {
    std::vector<int> data = {1, -1, 1, -1, 1, -1};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 0);
}

TEST(ReduceEdgeCasesTest, ZeroInit) {
    std::vector<int> data = {1, 2, 3};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 6);
}

TEST(ReduceEdgeCasesTest, OneInit) {
    std::vector<int> data = {2, 3, 4};
    int result = dispatch_reduce(seq{}, data.begin(), data.end(), 1, std::multiplies<>{});
    EXPECT_EQ(result, 24);
}

// =============================================================================
// Comparison: Sequential vs Parallel Consistency
// =============================================================================

TEST(ReduceConsistencyTest, SumConsistent) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 1);

    int seq_result = dispatch_reduce(seq{}, data.begin(), data.end(), 0, std::plus<>{});
    int par_result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});

    EXPECT_EQ(seq_result, par_result);
    EXPECT_EQ(seq_result, 500500);
}

TEST(ReduceConsistencyTest, ProductConsistent) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int seq_result = dispatch_reduce(seq{}, data.begin(), data.end(), 1, std::multiplies<>{});
    int par_result = dispatch_reduce(par{}, data.begin(), data.end(), 1, std::multiplies<>{});

    EXPECT_EQ(seq_result, par_result);
    EXPECT_EQ(seq_result, 3628800);  // 10!
}

TEST(ReduceConsistencyTest, MaxConsistent) {
    std::vector<int> data = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
    auto max_op = [](int a, int b) { return std::max(a, b); };

    int seq_result = dispatch_reduce(seq{}, data.begin(), data.end(),
                                      std::numeric_limits<int>::min(), max_op);
    int par_result = dispatch_reduce(par{}, data.begin(), data.end(),
                                      std::numeric_limits<int>::min(), max_op);

    EXPECT_EQ(seq_result, par_result);
    EXPECT_EQ(seq_result, 9);
}

// =============================================================================
// Associativity Tests (for parallel correctness)
// =============================================================================

TEST(AssociativityTest, AssociativeOperationWorks) {
    // std::plus is associative, so parallel should work correctly
    std::vector<int> data(10000);
    std::iota(data.begin(), data.end(), 1);

    int result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::plus<>{});
    EXPECT_EQ(result, 50005000);
}

TEST(AssociativityTest, XorAssociative) {
    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8};
    int result = dispatch_reduce(par{}, data.begin(), data.end(), 0, std::bit_xor<>{});
    // XOR is associative, result should be consistent
    EXPECT_EQ(result, 1 ^ 2 ^ 3 ^ 4 ^ 5 ^ 6 ^ 7 ^ 8);
}

}  // namespace dtl::test
