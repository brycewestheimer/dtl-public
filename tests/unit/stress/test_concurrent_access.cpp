// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_concurrent_access.cpp
/// @brief Stress tests for concurrent container access and memory pressure
/// @details Exercises concurrent read/write patterns, rapid container
///          creation/destruction, and mixed operation sequences.

#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/modifying/fill.hpp>
#include <dtl/algorithms/modifying/transform.hpp>
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/reductions/scan.hpp>
#include <dtl/algorithms/non_modifying/for_each.hpp>
#include <dtl/algorithms/non_modifying/count.hpp>
#include <dtl/algorithms/non_modifying/find.hpp>
#include <dtl/algorithms/non_modifying/predicates.hpp>
#include <dtl/algorithms/sorting/sort.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/partition/cyclic_partition.hpp>
#include <dtl/policies/partition/block_partition.hpp>
#include <dtl/error/result.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

namespace dtl::test {

// =============================================================================
// Concurrent Container Lifecycle
// =============================================================================

TEST(ConcurrentAccessTest, RapidCreateDestroyFromMultipleThreads) {
    // Multiple threads rapidly creating and destroying containers
    constexpr int num_threads = 4;
    constexpr int iterations_per_thread = 50;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, iterations_per_thread]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                // Each thread creates its own independent container
                distributed_vector<int> vec(1000 + t * 100, t * iterations_per_thread + i);
                auto local = vec.local_view();
                EXPECT_EQ(local.size(), static_cast<size_type>(1000 + t * 100));
                EXPECT_EQ(local[0], t * iterations_per_thread + i);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

TEST(ConcurrentAccessTest, IndependentContainerOperationsFromMultipleThreads) {
    // Each thread works on its own container independently
    constexpr int num_threads = 4;
    constexpr size_type N = 10'000;
    std::vector<std::thread> threads;
    std::vector<int64_t> results(num_threads, 0);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, N, &results]() {
            distributed_vector<int64_t> vec(N, static_cast<int64_t>(t + 1));

            // Fill, transform, reduce on independent containers
            auto fill_res = dtl::fill(seq{}, vec, static_cast<int64_t>(t + 1));
            ASSERT_TRUE(fill_res.has_value());

            results[t] = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Verify each thread got the right answer
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_EQ(results[t], static_cast<int64_t>((t + 1) * N))
            << "Thread " << t << " got wrong result";
    }
}

// =============================================================================
// Memory Pressure
// =============================================================================

TEST(MemoryPressureTest, ManySmallContainersAlive) {
    // Keep many small containers alive simultaneously
    // Use unique_ptr because distributed_vector is not movable (contains atomics)
    constexpr int count = 500;
    std::vector<std::unique_ptr<distributed_vector<int>>> containers;
    containers.reserve(count);

    for (int i = 0; i < count; ++i) {
        containers.push_back(std::make_unique<distributed_vector<int>>(100, i));
    }

    // Verify all containers are still valid
    for (int i = 0; i < count; ++i) {
        auto local = containers[i]->local_view();
        ASSERT_EQ(local.size(), 100u) << "Container " << i;
        EXPECT_EQ(local[0], i) << "Container " << i;
        EXPECT_EQ(local[99], i) << "Container " << i;
    }
}

TEST(MemoryPressureTest, GrowingSizeCreation) {
    // Create containers of increasing size
    for (size_type size = 1; size <= 1'000'000; size *= 10) {
        distributed_vector<int> vec(size, 42);
        auto local = vec.local_view();
        ASSERT_EQ(local.size(), size) << "Size " << size;
        EXPECT_EQ(local[0], 42) << "Size " << size;
        EXPECT_EQ(local[size - 1], 42) << "Size " << size;
    }
}

// =============================================================================
// Mixed Operation Sequences
// =============================================================================

TEST(MixedOperationsTest, AlternatingFillAndReduce) {
    constexpr size_type N = 100'000;
    distributed_vector<int64_t> vec(N);

    // Alternate between fill with different values and reduce
    for (int i = 1; i <= 100; ++i) {
        auto fill_res = dtl::fill(seq{}, vec, static_cast<int64_t>(i));
        ASSERT_TRUE(fill_res.has_value()) << "Fill failed at iteration " << i;

        int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
        EXPECT_EQ(sum, static_cast<int64_t>(i) * static_cast<int64_t>(N))
            << "Reduce mismatch at iteration " << i;
    }
}

TEST(MixedOperationsTest, TransformReduceChain) {
    constexpr size_type N = 50'000;
    distributed_vector<int64_t> vec(N, 1);
    distributed_vector<int64_t> temp(N, 0);

    // Chain: transform (x -> x+1), reduce, repeat
    for (int i = 0; i < 20; ++i) {
        auto res = dtl::transform(seq{}, vec, temp, [](int64_t x) { return x + 1; });
        ASSERT_TRUE(res.has_value()) << "Transform failed at iteration " << i;

        // Copy result back
        auto local_vec = vec.local_view();
        auto local_tmp = temp.local_view();
        for (size_type j = 0; j < N; ++j) {
            local_vec[j] = local_tmp[j];
        }

        int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
        EXPECT_EQ(sum, static_cast<int64_t>(i + 2) * static_cast<int64_t>(N))
            << "Chain mismatch at iteration " << i;
    }
}

// =============================================================================
// Edge Cases Under Stress
// =============================================================================

TEST(StressEdgeCaseTest, EmptyContainerOperations) {
    // Operations on zero-size containers should not crash
    distributed_vector<int> vec(0);
    auto local = vec.local_view();
    EXPECT_EQ(local.size(), 0u);

    int sum = dtl::reduce(seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(sum, 0);
}

TEST(StressEdgeCaseTest, SingleElementOperations) {
    constexpr int iterations = 1000;
    for (int i = 0; i < iterations; ++i) {
        distributed_vector<int> vec(1, i);
        auto local = vec.local_view();
        ASSERT_EQ(local.size(), 1u);
        EXPECT_EQ(local[0], i);

        int sum = dtl::reduce(seq{}, vec, 0, std::plus<>{});
        EXPECT_EQ(sum, i) << "Iteration " << i;
    }
}

TEST(StressEdgeCaseTest, AllSameValues) {
    // All elements the same - edge case for sort, unique, etc.
    constexpr size_type N = 100'000;
    distributed_vector<int> vec(N, 42);

    auto count_res = dtl::count(seq{}, vec, 42);
    EXPECT_EQ(count_res, static_cast<size_type>(N));
}

// =============================================================================
// Random Operation Sequences
// =============================================================================

TEST(RandomOperationsTest, RandomFillTransformReduceSequence) {
    constexpr size_type N = 10'000;
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(1, 100);

    distributed_vector<int64_t> vec(N);

    for (int iter = 0; iter < 50; ++iter) {
        int fill_val = dist(rng);
        auto fill_res = dtl::fill(seq{}, vec, static_cast<int64_t>(fill_val));
        ASSERT_TRUE(fill_res.has_value());

        int64_t sum = dtl::reduce(seq{}, vec, int64_t{0}, std::plus<>{});
        EXPECT_EQ(sum, static_cast<int64_t>(fill_val) * static_cast<int64_t>(N))
            << "Iteration " << iter << " with fill_val=" << fill_val;
    }
}

// =============================================================================
// Policy Variant Stress
// =============================================================================

TEST(PolicyStressTest, CyclicPartitionLargeScale) {
    constexpr size_type N = 100'000;
    distributed_vector<int, cyclic_partition<>> vec(N, 7);

    auto local = vec.local_view();
    ASSERT_EQ(local.size(), N);

    int sum = dtl::reduce(seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(sum, 7 * static_cast<int>(N));
}

TEST(PolicyStressTest, BlockPartitionLargeScale) {
    constexpr size_type N = 100'000;
    distributed_vector<int, block_partition<>> vec(N, 11);

    auto local = vec.local_view();
    ASSERT_EQ(local.size(), N);

    int sum = dtl::reduce(seq{}, vec, 0, std::plus<>{});
    EXPECT_EQ(sum, 11 * static_cast<int>(N));
}

}  // namespace dtl::test
