// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_work_stealing_pool.cpp
/// @brief Unit tests for work-stealing queue and pool

#include <backends/cpu/work_stealing_queue.hpp>
#include <backends/cpu/work_stealing_pool.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

namespace dtl::test {

// ============================================================================
// Work-Stealing Queue Tests
// ============================================================================

TEST(WorkStealingQueueTest, PushPop) {
    dtl::cpu::work_stealing_queue q;

    int result = 0;
    EXPECT_TRUE(q.push([&result] { result = 42; }));

    auto task = q.pop();
    ASSERT_TRUE(task.has_value());
    (*task)();
    EXPECT_EQ(result, 42);
}

TEST(WorkStealingQueueTest, FIFO_Steal) {
    dtl::cpu::work_stealing_queue q;

    std::vector<int> order;
    for (int i = 0; i < 5; ++i) {
        q.push([&order, i] { order.push_back(i); });
    }

    // Steal takes from the top (FIFO)
    for (int i = 0; i < 5; ++i) {
        auto task = q.steal();
        ASSERT_TRUE(task.has_value());
        (*task)();
    }
    ASSERT_EQ(order.size(), 5u);
    EXPECT_EQ(order[0], 0);
    EXPECT_EQ(order[4], 4);
}

TEST(WorkStealingQueueTest, LIFO_Pop) {
    dtl::cpu::work_stealing_queue q;

    std::vector<int> order;
    for (int i = 0; i < 5; ++i) {
        q.push([&order, i] { order.push_back(i); });
    }

    // Pop takes from the bottom (LIFO)
    for (int i = 0; i < 5; ++i) {
        auto task = q.pop();
        ASSERT_TRUE(task.has_value());
        (*task)();
    }
    ASSERT_EQ(order.size(), 5u);
    EXPECT_EQ(order[0], 4);  // Last pushed, first popped
    EXPECT_EQ(order[4], 0);
}

TEST(WorkStealingQueueTest, EmptyPop) {
    dtl::cpu::work_stealing_queue q;
    auto task = q.pop();
    EXPECT_FALSE(task.has_value());
}

TEST(WorkStealingQueueTest, EmptySteal) {
    dtl::cpu::work_stealing_queue q;
    auto task = q.steal();
    EXPECT_FALSE(task.has_value());
}

TEST(WorkStealingQueueTest, SizeAndEmpty) {
    dtl::cpu::work_stealing_queue q;
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    q.push([] {});
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 1u);

    q.pop();
    EXPECT_TRUE(q.empty());
}

// ============================================================================
// Work-Stealing Pool Tests
// ============================================================================

TEST(WorkStealingPoolTest, ConstructDestruct) {
    dtl::cpu::work_stealing_pool pool(2);
    EXPECT_EQ(pool.size(), 2u);
}

TEST(WorkStealingPoolTest, DefaultThreadCount) {
    dtl::cpu::work_stealing_pool pool;
    EXPECT_GE(pool.size(), 1u);
}

TEST(WorkStealingPoolTest, SingleTask) {
    dtl::cpu::work_stealing_pool pool(2);
    auto fut = pool.submit([]() { return 42; });
    EXPECT_EQ(fut.get(), 42);
}

TEST(WorkStealingPoolTest, MultipleTasksCorrectness) {
    dtl::cpu::work_stealing_pool pool(4);

    constexpr int N = 1000;
    std::vector<std::future<int>> futures;
    futures.reserve(N);

    for (int i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i]() { return i * i; }));
    }

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(futures[static_cast<size_t>(i)].get(), i * i);
    }
}

TEST(WorkStealingPoolTest, Wait) {
    dtl::cpu::work_stealing_pool pool(2);
    std::atomic<int> counter{0};

    for (int i = 0; i < 100; ++i) {
        pool.submit([&counter]() { counter.fetch_add(1); });
    }

    pool.wait();
    EXPECT_EQ(counter.load(), 100);
}

TEST(WorkStealingPoolTest, PendingApproximation) {
    dtl::cpu::work_stealing_pool pool(1);
    // With 1 thread, tasks might queue up
    // We can't test exact pending count due to timing,
    // but pending() should not crash
    auto p = pool.pending();
    (void)p;
    SUCCEED();
}

// ============================================================================
// Concurrent Stress Tests
// ============================================================================

TEST(WorkStealingPoolTest, ConcurrentSubmit) {
    dtl::cpu::work_stealing_pool pool(4);
    constexpr int tasks_per_thread = 500;
    constexpr int num_submitters = 4;
    std::atomic<int> total{0};

    std::vector<std::thread> submitters;
    submitters.reserve(num_submitters);
    for (int t = 0; t < num_submitters; ++t) {
        submitters.emplace_back([&pool, &total]() {
            for (int i = 0; i < tasks_per_thread; ++i) {
                pool.submit([&total]() { total.fetch_add(1); });
            }
        });
    }

    for (auto& s : submitters) {
        s.join();
    }
    pool.wait();

    EXPECT_EQ(total.load(), tasks_per_thread * num_submitters);
}

TEST(WorkStealingPoolTest, ConcurrentPoolStress) {
    // Stress test using the pool — many submitters, verify all tasks complete
    dtl::cpu::work_stealing_pool pool(4);
    constexpr int N = 10000;
    std::atomic<int> sum{0};

    // Multiple submitter threads push tasks concurrently
    constexpr int num_submitters = 4;
    constexpr int per_submitter = N / num_submitters;
    std::vector<std::thread> submitters;
    submitters.reserve(num_submitters);

    for (int s = 0; s < num_submitters; ++s) {
        int start = s * per_submitter;
        int end = (s == num_submitters - 1) ? N : (s + 1) * per_submitter;
        submitters.emplace_back([&pool, &sum, start, end]() {
            for (int i = start; i < end; ++i) {
                pool.submit([&sum, i]() { sum.fetch_add(i); });
            }
        });
    }

    for (auto& s : submitters) {
        s.join();
    }
    pool.wait();

    // Expected sum: 0 + 1 + ... + (N-1) = N*(N-1)/2
    EXPECT_EQ(sum.load(), N * (N - 1) / 2);
}

TEST(WorkStealingPoolTest, MixedWorkloads) {
    // Submit tasks with varying work amounts
    dtl::cpu::work_stealing_pool pool(4);
    constexpr int N = 200;
    std::atomic<long long> total{0};

    std::vector<std::future<long long>> futures;
    futures.reserve(N);

    for (int i = 0; i < N; ++i) {
        futures.push_back(pool.submit([i]() -> long long {
            long long sum = 0;
            // Varying workload
            for (int j = 0; j < (i % 10 + 1) * 100; ++j) {
                sum += j;
            }
            return sum;
        }));
    }

    long long grand_total = 0;
    for (auto& f : futures) {
        grand_total += f.get();
    }

    // Just verify all tasks completed without errors
    EXPECT_GT(grand_total, 0);
}

}  // namespace dtl::test
