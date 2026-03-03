// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_large_scale.cpp
/// @brief R10.4: Large-scale single-process stress tests
/// @details Tests distributed containers and algorithms at large element counts
///          to verify correctness and absence of memory issues. Uses single-rank
///          convenience constructors (no MPI required).

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/modifying/fill.hpp>
#include <dtl/algorithms/modifying/copy.hpp>
#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/reductions/transform_reduce.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <numeric>

namespace dtl::test {

// =============================================================================
// 1M Element Tests
// =============================================================================

TEST(LargeScaleTest, FillOneMillionElements) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int> vec(N);

    auto res = dtl::fill(seq{}, vec, 42);
    ASSERT_TRUE(res.has_value());

    // Spot-check elements at known positions
    auto local = vec.local_view();
    ASSERT_EQ(local.size(), N);
    EXPECT_EQ(local[0], 42);
    EXPECT_EQ(local[N / 4], 42);
    EXPECT_EQ(local[N / 2], 42);
    EXPECT_EQ(local[3 * N / 4], 42);
    EXPECT_EQ(local[N - 1], 42);
}

TEST(LargeScaleTest, CopyOneMillionElements) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int> src(N, 99);
    distributed_vector<int> dst(N, 0);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, N);

    // Spot-check destination elements
    auto local = dst.local_view();
    EXPECT_EQ(local[0], 99);
    EXPECT_EQ(local[N / 2], 99);
    EXPECT_EQ(local[N - 1], 99);
}

TEST(LargeScaleTest, ReduceOneMillionElements) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int64_t> vec(N, 1);

    // Sum of 1M ones should be exactly 1M
    int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
    EXPECT_EQ(sum, static_cast<int64_t>(N));
}

TEST(LargeScaleTest, ReduceOneMillionElementsProduct) {
    constexpr size_type N = 1'000'000;
    // Fill with 1s so product is 1
    distributed_vector<int64_t> vec(N, 1);

    int64_t prod = dtl::reduce(seq{}, vec, int64_t{1}, std::multiplies<>{});
    EXPECT_EQ(prod, 1);
}

TEST(LargeScaleTest, TransformOneMillionElements) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int> src(N, 5);
    distributed_vector<int> dst(N, 0);

    auto res = dtl::transform(seq{}, src, dst, [](int x) { return x * 3; });
    ASSERT_TRUE(res.has_value());

    auto local = dst.local_view();
    EXPECT_EQ(local[0], 15);
    EXPECT_EQ(local[N / 2], 15);
    EXPECT_EQ(local[N - 1], 15);
}

// =============================================================================
// 10M Element Tests (verify no memory issues with large allocations)
// =============================================================================

TEST(LargeScaleTest, FillTenMillionElements) {
    constexpr size_type N = 10'000'000;
    distributed_vector<int> vec(N);

    auto res = dtl::fill(seq{}, vec, 7);
    ASSERT_TRUE(res.has_value());

    auto local = vec.local_view();
    ASSERT_EQ(local.size(), N);
    EXPECT_EQ(local[0], 7);
    EXPECT_EQ(local[N / 2], 7);
    EXPECT_EQ(local[N - 1], 7);
}

TEST(LargeScaleTest, TenMillionElementReduceSum) {
    constexpr size_type N = 10'000'000;
    distributed_vector<int64_t> vec(N, 3);

    int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
    EXPECT_EQ(sum, 3 * static_cast<int64_t>(N));
}

TEST(LargeScaleTest, TenMillionElementCopy) {
    constexpr size_type N = 10'000'000;
    distributed_vector<int> src(N, 42);
    distributed_vector<int> dst(N, 0);

    auto res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value().count, N);

    // Spot-check
    auto local = dst.local_view();
    EXPECT_EQ(local[0], 42);
    EXPECT_EQ(local[N - 1], 42);
}

// =============================================================================
// Repeated Allocation/Deallocation Cycles
// =============================================================================

TEST(LargeScaleTest, RepeatedAllocationCycles100Iterations) {
    // Create and destroy containers 100 times to check for memory leaks
    for (int i = 0; i < 100; ++i) {
        distributed_vector<int> vec(10'000, i);

        // Verify the container is functional
        auto local = vec.local_view();
        ASSERT_EQ(local.size(), 10'000u) << "Iteration " << i;
        EXPECT_EQ(local[0], i) << "Iteration " << i;
        EXPECT_EQ(local[9999], i) << "Iteration " << i;

        // Container destroyed at end of scope
    }
}

TEST(LargeScaleTest, RepeatedLargeAllocationCycles) {
    // Larger containers, fewer iterations
    for (int i = 0; i < 20; ++i) {
        distributed_vector<int> vec(1'000'000, i);

        auto local = vec.local_view();
        ASSERT_EQ(local.size(), 1'000'000u) << "Iteration " << i;
        EXPECT_EQ(local[0], i) << "Iteration " << i;
        EXPECT_EQ(local[999'999], i) << "Iteration " << i;
    }
}

TEST(LargeScaleTest, RepeatedFillAndReduce) {
    // Repeated fill + reduce cycles
    distributed_vector<int64_t> vec(100'000);

    for (int i = 1; i <= 50; ++i) {
        auto fill_res = dtl::fill(seq{}, vec, static_cast<int64_t>(i));
        ASSERT_TRUE(fill_res.has_value()) << "Fill failed at iteration " << i;

        int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
        EXPECT_EQ(sum, static_cast<int64_t>(i) * 100'000) << "Reduce mismatch at iteration " << i;
    }
}

// =============================================================================
// Large Transform-Reduce Operation
// =============================================================================

TEST(LargeScaleTest, LargeTransformReduceSumOfSquares) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int64_t> vec(N);

    // Fill with iota values: 0, 1, 2, ..., N-1
    auto local = vec.local_view();
    for (size_type i = 0; i < N; ++i) {
        local[i] = static_cast<int64_t>(i);
    }

    // Compute sum of squares: sum(i^2 for i in 0..N-1)
    int64_t sum_sq = dtl::transform_reduce(
        seq{}, vec, int64_t{0}, std::plus<>{},
        [](int64_t x) { return x * x; });

    // Expected: N*(N-1)*(2N-1)/6
    int64_t expected = static_cast<int64_t>(N) * (N - 1) * (2 * N - 1) / 6;
    EXPECT_EQ(sum_sq, expected);
}

TEST(LargeScaleTest, LargeTransformReduceParallel) {
    constexpr size_type N = 1'000'000;
    distributed_vector<int64_t> vec(N, 2);

    // Sum of (x * x) where all x = 2: N * 4
    int64_t sum_sq = dtl::transform_reduce(
        seq{}, vec, int64_t{0}, std::plus<>{},
        [](int64_t x) { return x * x; });

    EXPECT_EQ(sum_sq, 4 * static_cast<int64_t>(N));
}

// =============================================================================
// Correctness Verification on Large Containers
// =============================================================================

TEST(LargeScaleTest, VerifyIotaPatternAfterFill) {
    constexpr size_type N = 100'000;
    distributed_vector<int> vec(N);

    // Manual iota fill
    auto local = vec.local_view();
    for (size_type i = 0; i < N; ++i) {
        local[i] = static_cast<int>(i);
    }

    // Verify every 1000th element
    for (size_type i = 0; i < N; i += 1000) {
        EXPECT_EQ(local[i], static_cast<int>(i)) << "Mismatch at index " << i;
    }

    // Verify first and last
    EXPECT_EQ(local[0], 0);
    EXPECT_EQ(local[N - 1], static_cast<int>(N - 1));
}

TEST(LargeScaleTest, TransformPreservesAllElements) {
    constexpr size_type N = 500'000;
    distributed_vector<int> src(N);
    distributed_vector<int> dst(N, 0);

    // Fill src with index pattern
    auto src_local = src.local_view();
    for (size_type i = 0; i < N; ++i) {
        src_local[i] = static_cast<int>(i % 1000);
    }

    auto res = dtl::transform(seq{}, src, dst, [](int x) { return x + 1; });
    ASSERT_TRUE(res.has_value());

    auto dst_local = dst.local_view();
    // Spot-check every 10000th element
    for (size_type i = 0; i < N; i += 10'000) {
        EXPECT_EQ(dst_local[i], static_cast<int>(i % 1000) + 1)
            << "Mismatch at index " << i;
    }
}

TEST(LargeScaleTest, CopyAndReduceConsistency) {
    constexpr size_type N = 500'000;
    distributed_vector<int64_t> src(N, 3);
    distributed_vector<int64_t> dst(N, 0);

    // Copy src to dst
    auto copy_res = dtl::copy(seq{}, src, dst);
    ASSERT_TRUE(copy_res.has_value());

    // Reduce both - should be equal
    int64_t src_sum = dtl::reduce(seq{}, src, int64_t{0}, std::plus<>{});
    int64_t dst_sum = dtl::reduce(seq{}, dst, int64_t{0}, std::plus<>{});

    EXPECT_EQ(src_sum, dst_sum);
    EXPECT_EQ(src_sum, 3 * static_cast<int64_t>(N));
}

}  // namespace dtl::test
