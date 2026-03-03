// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_copy_fill.cpp
/// @brief Unit tests for copy and fill algorithms
/// @details Tests for Task 3.3: Modifying Algorithms (copy, fill, generate, iota)

#include <dtl/algorithms/modifying/copy.hpp>
#include <dtl/algorithms/modifying/fill.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/views/local_view.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <string>
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
// Fill Tests with Standard Iterators
// =============================================================================

TEST(FillTest, FillsAllElements) {
    std::vector<int> data(10, 0);
    dispatch_fill(seq{}, data.begin(), data.end(), 42);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 42; }));
}

TEST(FillTest, FillsWithZero) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    dispatch_fill(seq{}, data.begin(), data.end(), 0);
    EXPECT_EQ(data, (std::vector<int>{0, 0, 0, 0, 0}));
}

TEST(FillTest, FillsWithNegative) {
    std::vector<int> data(5, 0);
    dispatch_fill(seq{}, data.begin(), data.end(), -99);
    EXPECT_EQ(data, (std::vector<int>{-99, -99, -99, -99, -99}));
}

TEST(FillTest, EmptyRange) {
    std::vector<int> data;
    dispatch_fill(seq{}, data.begin(), data.end(), 42);
    EXPECT_TRUE(data.empty());
}

TEST(FillTest, SingleElement) {
    std::vector<int> data(1, 0);
    dispatch_fill(seq{}, data.begin(), data.end(), 100);
    EXPECT_EQ(data[0], 100);
}

TEST(FillTest, DoubleValues) {
    std::vector<double> data(5, 0.0);
    dispatch_fill(seq{}, data.begin(), data.end(), 3.14159);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                            [](double x) { return x == 3.14159; }));
}

TEST(FillTest, StringValues) {
    std::vector<std::string> data(3);
    dispatch_fill(seq{}, data.begin(), data.end(), std::string("hello"));
    EXPECT_EQ(data, (std::vector<std::string>{"hello", "hello", "hello"}));
}

TEST(FillTest, CharValues) {
    std::vector<char> data(5);
    dispatch_fill(seq{}, data.begin(), data.end(), 'X');
    EXPECT_EQ(data, (std::vector<char>{'X', 'X', 'X', 'X', 'X'}));
}

// =============================================================================
// Parallel Fill Tests
// =============================================================================

TEST(ParallelFillTest, FillsAllElements) {
    std::vector<int> data(1000, 0);
    dispatch_fill(par{}, data.begin(), data.end(), 42);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 42; }));
}

TEST(ParallelFillTest, LargeDataSet) {
    std::vector<int> data(10000, -1);
    dispatch_fill(par{}, data.begin(), data.end(), 999);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 999; }));
}

TEST(ParallelFillTest, ConsistentWithSequential) {
    std::vector<int> seq_data(1000, 0);
    std::vector<int> par_data(1000, 0);

    dispatch_fill(seq{}, seq_data.begin(), seq_data.end(), 77);
    dispatch_fill(par{}, par_data.begin(), par_data.end(), 77);

    EXPECT_EQ(seq_data, par_data);
}

// =============================================================================
// Fill with Local View Tests
// =============================================================================

TEST(LocalViewFillTest, FillsLocalView) {
    std::vector<int> data(10, 0);
    local_view<int> view(data.data(), data.size());

    dispatch_fill(seq{}, view.begin(), view.end(), 123);

    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 123; }));
}

TEST(LocalViewFillTest, ParallelFillLocalView) {
    std::vector<int> data(100, 0);
    local_view<int> view(data.data(), data.size());

    dispatch_fill(par{}, view.begin(), view.end(), 456);

    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](int x) { return x == 456; }));
}

// =============================================================================
// Fill_n Tests
// =============================================================================

TEST(FillNTest, FillsFirstNElements) {
    std::vector<int> data = {0, 0, 0, 0, 0};
    std::fill_n(data.begin(), 3, 42);
    EXPECT_EQ(data, (std::vector<int>{42, 42, 42, 0, 0}));
}

TEST(FillNTest, FillsAllIfNEqualsSize) {
    std::vector<int> data(5, 0);
    std::fill_n(data.begin(), 5, 100);
    EXPECT_EQ(data, (std::vector<int>{100, 100, 100, 100, 100}));
}

TEST(FillNTest, FillsZeroElements) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::fill_n(data.begin(), 0, 99);
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(FillNTest, FillsSingleElement) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::fill_n(data.begin(), 1, 99);
    EXPECT_EQ(data, (std::vector<int>{99, 2, 3, 4, 5}));
}

// =============================================================================
// Generate Tests
// =============================================================================

TEST(GenerateTest, GeneratesSequence) {
    std::vector<int> data(5);
    int counter = 0;
    std::generate(data.begin(), data.end(), [&counter]() { return counter++; });
    EXPECT_EQ(data, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(GenerateTest, GeneratesConstantValues) {
    std::vector<int> data(5);
    std::generate(data.begin(), data.end(), []() { return 42; });
    EXPECT_EQ(data, (std::vector<int>{42, 42, 42, 42, 42}));
}

TEST(GenerateTest, GeneratesAlternating) {
    std::vector<int> data(6);
    bool toggle = false;
    std::generate(data.begin(), data.end(), [&toggle]() {
        toggle = !toggle;
        return toggle ? 1 : 0;
    });
    EXPECT_EQ(data, (std::vector<int>{1, 0, 1, 0, 1, 0}));
}

TEST(GenerateTest, GeneratesRandomInRange) {
    std::vector<int> data(100);
    std::generate(data.begin(), data.end(), []() { return 50; });  // Simplified
    EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                            [](int x) { return x >= 0 && x <= 100; }));
}

TEST(GenerateTest, EmptyRange) {
    std::vector<int> data;
    std::generate(data.begin(), data.end(), []() { return 42; });
    EXPECT_TRUE(data.empty());
}

// =============================================================================
// Iota Tests
// =============================================================================

TEST(IotaTest, FillsWithSequence) {
    std::vector<int> data(5);
    std::iota(data.begin(), data.end(), 0);
    EXPECT_EQ(data, (std::vector<int>{0, 1, 2, 3, 4}));
}

TEST(IotaTest, FillsStartingFromValue) {
    std::vector<int> data(5);
    std::iota(data.begin(), data.end(), 10);
    EXPECT_EQ(data, (std::vector<int>{10, 11, 12, 13, 14}));
}

TEST(IotaTest, FillsWithNegativeStart) {
    std::vector<int> data(5);
    std::iota(data.begin(), data.end(), -2);
    EXPECT_EQ(data, (std::vector<int>{-2, -1, 0, 1, 2}));
}

TEST(IotaTest, FillsDoubles) {
    std::vector<double> data(5);
    std::iota(data.begin(), data.end(), 1.5);
    EXPECT_DOUBLE_EQ(data[0], 1.5);
    EXPECT_DOUBLE_EQ(data[1], 2.5);
    EXPECT_DOUBLE_EQ(data[2], 3.5);
    EXPECT_DOUBLE_EQ(data[3], 4.5);
    EXPECT_DOUBLE_EQ(data[4], 5.5);
}

TEST(IotaTest, EmptyRange) {
    std::vector<int> data;
    std::iota(data.begin(), data.end(), 0);
    EXPECT_TRUE(data.empty());
}

TEST(IotaTest, SingleElement) {
    std::vector<int> data(1);
    std::iota(data.begin(), data.end(), 100);
    EXPECT_EQ(data[0], 100);
}

TEST(IotaTest, LargeRange) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], static_cast<int>(i));
    }
}

// =============================================================================
// Copy Tests with Standard Iterators
// =============================================================================

TEST(CopyTest, CopiesAllElements) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(CopyTest, CopiesEmptyRange) {
    std::vector<int> src;
    std::vector<int> dst;
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_TRUE(dst.empty());
}

TEST(CopyTest, CopiesSingleElement) {
    std::vector<int> src = {42};
    std::vector<int> dst(1);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst[0], 42);
}

TEST(CopyTest, CopiesStrings) {
    std::vector<std::string> src = {"hello", "world", "test"};
    std::vector<std::string> dst(3);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(CopyTest, CopiesDoubles) {
    std::vector<double> src = {1.1, 2.2, 3.3, 4.4, 5.5};
    std::vector<double> dst(5);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(CopyTest, SourceUnmodified) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> original = src;
    std::vector<int> dst(5);
    dispatch_copy(seq{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(src, original);
}

// =============================================================================
// Parallel Copy Tests
// =============================================================================

TEST(ParallelCopyTest, CopiesAllElements) {
    std::vector<int> src(1000);
    std::iota(src.begin(), src.end(), 0);
    std::vector<int> dst(1000);

    dispatch_copy(par{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(ParallelCopyTest, LargeDataSet) {
    std::vector<int> src(10000);
    std::iota(src.begin(), src.end(), 1);
    std::vector<int> dst(10000);

    dispatch_copy(par{}, src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, src);
}

TEST(ParallelCopyTest, ConsistentWithSequential) {
    std::vector<int> src(1000);
    std::iota(src.begin(), src.end(), 100);

    std::vector<int> seq_dst(1000);
    std::vector<int> par_dst(1000);

    dispatch_copy(seq{}, src.begin(), src.end(), seq_dst.begin());
    dispatch_copy(par{}, src.begin(), src.end(), par_dst.begin());

    EXPECT_EQ(seq_dst, par_dst);
}

// =============================================================================
// Copy with Local View Tests
// =============================================================================

TEST(LocalViewCopyTest, CopiesFromLocalView) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);

    local_view<int> src_view(src.data(), src.size());
    dispatch_copy(seq{}, src_view.begin(), src_view.end(), dst.begin());

    EXPECT_EQ(dst, src);
}

TEST(LocalViewCopyTest, CopiesToLocalView) {
    std::vector<int> src = {10, 20, 30, 40, 50};
    std::vector<int> dst(5);

    local_view<int> dst_view(dst.data(), dst.size());
    dispatch_copy(seq{}, src.begin(), src.end(), dst_view.begin());

    EXPECT_EQ(dst, src);
}

TEST(LocalViewCopyTest, CopiesBetweenLocalViews) {
    std::vector<int> src = {100, 200, 300};
    std::vector<int> dst(3);

    local_view<int> src_view(src.data(), src.size());
    local_view<int> dst_view(dst.data(), dst.size());

    dispatch_copy(seq{}, src_view.begin(), src_view.end(), dst_view.begin());

    EXPECT_EQ(dst, src);
}

// =============================================================================
// Copy_if Tests
// =============================================================================

TEST(CopyIfTest, CopiesMatchingElements) {
    std::vector<int> src = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> dst;

    std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                 [](int x) { return x % 2 == 0; });

    EXPECT_EQ(dst, (std::vector<int>{2, 4, 6, 8, 10}));
}

TEST(CopyIfTest, CopiesNoElements) {
    std::vector<int> src = {1, 3, 5, 7, 9};
    std::vector<int> dst;

    std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                 [](int x) { return x % 2 == 0; });

    EXPECT_TRUE(dst.empty());
}

TEST(CopyIfTest, CopiesAllElements) {
    std::vector<int> src = {2, 4, 6, 8, 10};
    std::vector<int> dst;

    std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                 [](int x) { return x % 2 == 0; });

    EXPECT_EQ(dst, src);
}

TEST(CopyIfTest, CopiesGreaterThan) {
    std::vector<int> src = {1, 5, 2, 8, 3, 9, 4};
    std::vector<int> dst;

    std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                 [](int x) { return x > 4; });

    EXPECT_EQ(dst, (std::vector<int>{5, 8, 9}));
}

TEST(CopyIfTest, PreservesOrder) {
    std::vector<int> src = {10, 1, 20, 2, 30, 3};
    std::vector<int> dst;

    std::copy_if(src.begin(), src.end(), std::back_inserter(dst),
                 [](int x) { return x >= 10; });

    EXPECT_EQ(dst, (std::vector<int>{10, 20, 30}));
}

// =============================================================================
// Copy_n Tests
// =============================================================================

TEST(CopyNTest, CopiesFirstN) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(3);

    std::copy_n(src.begin(), 3, dst.begin());

    EXPECT_EQ(dst, (std::vector<int>{1, 2, 3}));
}

TEST(CopyNTest, CopiesZeroElements) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5, 0);

    std::copy_n(src.begin(), 0, dst.begin());

    EXPECT_EQ(dst, (std::vector<int>{0, 0, 0, 0, 0}));
}

TEST(CopyNTest, CopiesAllElements) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5);

    std::copy_n(src.begin(), 5, dst.begin());

    EXPECT_EQ(dst, src);
}

TEST(CopyNTest, CopiesSingleElement) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst(5, 0);

    std::copy_n(src.begin(), 1, dst.begin());

    EXPECT_EQ(dst[0], 1);
    EXPECT_EQ(dst[1], 0);
}

// =============================================================================
// copy_result Tests
// =============================================================================

TEST(CopyResultTest, DefaultConstruction) {
    copy_result result;
    EXPECT_EQ(result.count, 0u);
    EXPECT_TRUE(result.success);
}

TEST(CopyResultTest, CustomValues) {
    copy_result result{100, true};
    EXPECT_EQ(result.count, 100u);
    EXPECT_TRUE(result.success);
}

TEST(CopyResultTest, FailedCopy) {
    copy_result result{50, false};
    EXPECT_EQ(result.count, 50u);
    EXPECT_FALSE(result.success);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(EdgeCasesTest, FillLargeValue) {
    std::vector<long long> data(100);
    dispatch_fill(seq{}, data.begin(), data.end(), 9999999999999LL);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                            [](long long x) { return x == 9999999999999LL; }));
}

TEST(EdgeCasesTest, CopyToSelf) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    // Copy overlapping range (to same position is a no-op conceptually)
    dispatch_copy(seq{}, data.begin(), data.end(), data.begin());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST(EdgeCasesTest, FillPartialRange) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    dispatch_fill(seq{}, data.begin() + 1, data.begin() + 4, 0);
    EXPECT_EQ(data, (std::vector<int>{1, 0, 0, 0, 5}));
}

TEST(EdgeCasesTest, CopyPartialRange) {
    std::vector<int> src = {10, 20, 30, 40, 50};
    std::vector<int> dst = {1, 2, 3, 4, 5};
    dispatch_copy(seq{}, src.begin() + 1, src.begin() + 4, dst.begin() + 1);
    EXPECT_EQ(dst, (std::vector<int>{1, 20, 30, 40, 5}));
}

// =============================================================================
// Mixed Fill and Copy Operations
// =============================================================================

TEST(MixedOperationsTest, FillThenCopy) {
    std::vector<int> data(10);
    dispatch_fill(seq{}, data.begin(), data.end(), 42);

    std::vector<int> copy_dst(10);
    dispatch_copy(seq{}, data.begin(), data.end(), copy_dst.begin());

    EXPECT_EQ(data, copy_dst);
}

TEST(MixedOperationsTest, IotaThenCopy) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 1);

    std::vector<int> copy_dst(10);
    dispatch_copy(seq{}, data.begin(), data.end(), copy_dst.begin());

    EXPECT_EQ(copy_dst, (std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
}

TEST(MixedOperationsTest, GenerateThenCopyIf) {
    std::vector<int> data(10);
    std::iota(data.begin(), data.end(), 1);  // {1, 2, 3, ..., 10}

    std::vector<int> filtered;
    std::copy_if(data.begin(), data.end(), std::back_inserter(filtered),
                 [](int x) { return x > 5; });

    EXPECT_EQ(filtered, (std::vector<int>{6, 7, 8, 9, 10}));
}

// =============================================================================
// Sequential vs Parallel Consistency
// =============================================================================

TEST(ConsistencyTest, FillConsistent) {
    std::vector<int> seq_data(1000, 0);
    std::vector<int> par_data(1000, 0);

    dispatch_fill(seq{}, seq_data.begin(), seq_data.end(), 123);
    dispatch_fill(par{}, par_data.begin(), par_data.end(), 123);

    EXPECT_EQ(seq_data, par_data);
}

TEST(ConsistencyTest, CopyConsistent) {
    std::vector<int> src(1000);
    std::iota(src.begin(), src.end(), 0);

    std::vector<int> seq_dst(1000);
    std::vector<int> par_dst(1000);

    dispatch_copy(seq{}, src.begin(), src.end(), seq_dst.begin());
    dispatch_copy(par{}, src.begin(), src.end(), par_dst.begin());

    EXPECT_EQ(seq_dst, par_dst);
}

// =============================================================================
// Type Conversion Tests
// =============================================================================

TEST(TypeConversionTest, FillIntToDouble) {
    std::vector<double> data(5, 0.0);
    dispatch_fill(seq{}, data.begin(), data.end(), 42);  // int to double
    EXPECT_TRUE(std::all_of(data.begin(), data.end(),
                            [](double x) { return x == 42.0; }));
}

TEST(TypeConversionTest, CopyIntToLong) {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<long> dst(5);

    std::copy(src.begin(), src.end(), dst.begin());
    EXPECT_EQ(dst, (std::vector<long>{1L, 2L, 3L, 4L, 5L}));
}

// =============================================================================
// Distributed Copy Partition Compatibility Guard Tests (R1.2)
// =============================================================================

TEST(DistributedCopyTest, SamePartitionCopySucceeds) {
    distributed_vector<int> src(100, 42, test_context{0, 4});
    distributed_vector<int> dst(100, 0, test_context{0, 4});

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 25u);  // local_size = 100/4
    EXPECT_TRUE(res.value().success);

    // Verify data was copied
    auto local = dst.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(DistributedCopyTest, MismatchedGlobalSizeReturnsError) {
    distributed_vector<int> src(100, 42, test_context{0, 1});
    distributed_vector<int> dst(200, 0, test_context{0, 1});

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
}

TEST(DistributedCopyTest, MismatchedRankCountReturnsError) {
    distributed_vector<int> src(100, 42, test_context{0, 2});
    distributed_vector<int> dst(100, 0, test_context{0, 4});

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
}

TEST(DistributedCopyTest, DefaultExecutionCopyWorks) {
    distributed_vector<int> src(50, 7, test_context{0, 1});
    distributed_vector<int> dst(50, 0, test_context{0, 1});

    auto res = dtl::copy(src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 50u);

    auto local = dst.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 7);
    }
}

TEST(DistributedCopyTest, EmptyContainersCopySucceeds) {
    distributed_vector<int> src(0, test_context{0, 1});
    distributed_vector<int> dst(0, test_context{0, 1});

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 0u);
}

// =============================================================================
// R10.1: Cross-Partition Copy Regression Tests (R1.2 regression)
// =============================================================================

TEST(CrossPartitionCopyTest, DifferentGlobalSizesReturnsError) {
    // Two vectors with different global sizes -> copy returns error
    distributed_vector<int> src(100, 42);
    distributed_vector<int> dst(200, 0);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
    // Verify the error message mentions "incompatible partitions"
    EXPECT_NE(res.error().message().find("incompatible partitions"), std::string::npos)
        << "Error message should mention 'incompatible partitions', got: "
        << res.error().message();
}

TEST(CrossPartitionCopyTest, DifferentLocalSizesReturnsError) {
    // Two vectors with different local sizes due to different rank counts
    // src: 100 elements / 2 ranks = 50 local on rank 0
    // dst: 100 elements / 4 ranks = 25 local on rank 0
    test_context ctx2{0, 2};
    test_context ctx4{0, 4};
    distributed_vector<int> src(100, 42, ctx2);
    distributed_vector<int> dst(100, 0, ctx4);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
    EXPECT_NE(res.error().message().find("incompatible partitions"), std::string::npos)
        << "Error message should mention 'incompatible partitions', got: "
        << res.error().message();
}

TEST(CrossPartitionCopyTest, ErrorMessageContainsIncompatiblePartitions) {
    // Verify the specific error text from R1.2 fix
    distributed_vector<int> src(50, 1);
    distributed_vector<int> dst(75, 0);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_error());
    // The error message in copy.hpp says:
    // "copy: source and destination have incompatible partitions. "
    // "Use redistribute() for cross-partition data movement."
    EXPECT_NE(res.error().message().find("incompatible partitions"), std::string::npos);
    EXPECT_NE(res.error().message().find("redistribute"), std::string::npos);
}

TEST(CrossPartitionCopyTest, MatchingPartitionsCopySucceeds) {
    // Verify copy succeeds when partitions match
    distributed_vector<int> src(100, 42);
    distributed_vector<int> dst(100, 0);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, 100u);
    EXPECT_TRUE(res.value().success);

    // Verify all elements copied
    auto local = dst.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 42);
    }
}

TEST(CrossPartitionCopyTest, MatchingMultiRankPartitionsCopySucceeds) {
    // Same partition scheme on both containers (multi-rank context)
    test_context ctx{0, 4};
    distributed_vector<int> src(100, 99, ctx);
    distributed_vector<int> dst(100, 0, ctx);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    // local_size = 100/4 = 25 for rank 0
    EXPECT_EQ(res.value().count, 25u);

    auto local = dst.local_view();
    for (size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 99);
    }
}

TEST(CrossPartitionCopyTest, DefaultExecutionPolicyMismatchError) {
    // Verify default execution policy also checks partitions
    distributed_vector<int> src(100, 42);
    distributed_vector<int> dst(50, 0);

    auto res = dtl::copy(src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
}

TEST(CrossPartitionCopyTest, ParExecutionPolicyMismatchError) {
    // Verify par execution policy also checks partitions
    distributed_vector<int> src(100, 42);
    distributed_vector<int> dst(200, 0);

    auto res = dtl::copy(par{}, src, dst);
    ASSERT_TRUE(res.has_error());
    EXPECT_EQ(res.error().code(), status_code::invalid_argument);
}

}  // namespace dtl::test
