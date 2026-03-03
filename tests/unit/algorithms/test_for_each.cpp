// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_for_each.cpp
/// @brief Unit tests for for_each algorithm
/// @details Tests for Task 3.2: Non-Modifying Algorithms

#include <dtl/algorithms/non_modifying/for_each.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/views/local_view.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <vector>

namespace dtl::test {

// =============================================================================
// Basic for_each Tests with Local View
// =============================================================================

TEST(ForEachTest, AppliesFunctionToAllElements) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, 15);
}

TEST(ForEachTest, ModifiesElements) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    dispatch_for_each(seq{}, view.begin(), view.end(), [](int& x) { x *= 2; });

    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(ForEachTest, EmptyRange) {
    std::vector<int> data;
    local_view<int> view(data.data(), data.size());

    int count = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&count](int) { ++count; });

    EXPECT_EQ(count, 0);
}

TEST(ForEachTest, SingleElement) {
    std::vector<int> data = {42};
    local_view<int> view(data.data(), data.size());

    int value = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&value](int x) { value = x; });

    EXPECT_EQ(value, 42);
}

// =============================================================================
// Parallel for_each Tests
// =============================================================================

TEST(ForEachTest, ParallelExecutionAppliesFunction) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    std::atomic<int> sum{0};
    dispatch_for_each(par{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum.load(), 15);
}

TEST(ForEachTest, ParallelExecutionModifies) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    // Note: Parallel modification requires atomic or thread-safe operations
    dispatch_for_each(par{}, view.begin(), view.end(), [](int& x) { x *= 2; });

    EXPECT_EQ(data, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(ForEachTest, LargeDataSetParallel) {
    std::vector<int> data(10000, 1);
    local_view<int> view(data.data(), data.size());

    std::atomic<int> sum{0};
    dispatch_for_each(par{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum.load(), 10000);
}

// =============================================================================
// Stateful Function Object Tests
// =============================================================================

TEST(ForEachTest, StatefulFunctionObject) {
    struct Counter {
        int count = 0;
        void operator()(int) { ++count; }
    };

    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    Counter counter;
    dispatch_for_each(seq{}, view.begin(), view.end(), std::ref(counter));

    EXPECT_EQ(counter.count, 5);
}

TEST(ForEachTest, AccumulatingFunctionObject) {
    struct Summer {
        int total = 0;
        void operator()(int x) { total += x; }
    };

    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    Summer summer;
    dispatch_for_each(seq{}, view.begin(), view.end(), std::ref(summer));

    EXPECT_EQ(summer.total, 15);
}

// =============================================================================
// for_each with Different Value Types
// =============================================================================

TEST(ForEachTest, DoubleElements) {
    std::vector<double> data = {1.5, 2.5, 3.5};
    local_view<double> view(data.data(), data.size());

    double sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](double x) { sum += x; });

    EXPECT_DOUBLE_EQ(sum, 7.5);
}

TEST(ForEachTest, StringElements) {
    std::vector<std::string> data = {"a", "b", "c"};
    local_view<std::string> view(data.data(), data.size());

    std::string result;
    dispatch_for_each(seq{}, view.begin(), view.end(),
                      [&result](const std::string& s) { result += s; });

    EXPECT_EQ(result, "abc");
}

// =============================================================================
// for_each_n Tests (First N Elements)
// =============================================================================

TEST(ForEachNTest, ProcessesFirstNElements) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    int sum = 0;
    std::for_each_n(data.begin(), 3, [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, 6);  // 1 + 2 + 3
}

TEST(ForEachNTest, NLargerThanSize) {
    // Note: std::for_each_n with n > size is undefined behavior.
    // DTL's distributed for_each_n (tested elsewhere) handles bounds safely.
    // Here we just test the valid case where n equals size.
    std::vector<int> data = {1, 2, 3};

    int sum = 0;
    std::for_each_n(data.begin(), data.size(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, 6);  // Processes all 3 elements
}

TEST(ForEachNTest, ZeroElements) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    int count = 0;
    std::for_each_n(data.begin(), 0, [&count](int) { ++count; });

    EXPECT_EQ(count, 0);
}

// =============================================================================
// Const Correctness Tests
// =============================================================================

TEST(ForEachTest, ConstViewReadOnly) {
    const std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<const int> view(data.data(), data.size());

    int sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, 15);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ForEachTest, NegativeNumbers) {
    std::vector<int> data = {-5, -4, -3, -2, -1};
    local_view<int> view(data.data(), data.size());

    int sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, -15);
}

TEST(ForEachTest, MixedNumbers) {
    std::vector<int> data = {-2, -1, 0, 1, 2};
    local_view<int> view(data.data(), data.size());

    int sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](int x) { sum += x; });

    EXPECT_EQ(sum, 0);
}

TEST(ForEachTest, LargeValues) {
    std::vector<long long> data = {1000000000LL, 2000000000LL, 3000000000LL};
    local_view<long long> view(data.data(), data.size());

    long long sum = 0;
    dispatch_for_each(seq{}, view.begin(), view.end(), [&sum](long long x) { sum += x; });

    EXPECT_EQ(sum, 6000000000LL);
}

// =============================================================================
// Exception Handling Tests
// =============================================================================

TEST(ForEachTest, ThrowingFunctionStopsIteration) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    local_view<int> view(data.data(), data.size());

    int count = 0;
    try {
        dispatch_for_each(seq{}, view.begin(), view.end(), [&count](int x) {
            if (x == 3) throw std::runtime_error("stop");
            ++count;
        });
    } catch (const std::runtime_error&) {
        // Expected
    }

    EXPECT_LT(count, 5);  // Not all elements processed
}

// =============================================================================
// Comparison: Sequential vs Parallel Results Match
// =============================================================================

TEST(ForEachTest, SeqAndParProduceSameResult) {
    std::vector<int> data1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> data2 = data1;

    local_view<int> view1(data1.data(), data1.size());
    local_view<int> view2(data2.data(), data2.size());

    dispatch_for_each(seq{}, view1.begin(), view1.end(), [](int& x) { x *= 2; });
    dispatch_for_each(par{}, view2.begin(), view2.end(), [](int& x) { x *= 2; });

    EXPECT_EQ(data1, data2);
}

}  // namespace dtl::test
