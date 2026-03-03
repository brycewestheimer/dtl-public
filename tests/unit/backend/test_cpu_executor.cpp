// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cpu_executor.cpp
/// @brief Unit tests for CPU executor
/// @details Verifies cpu_executor satisfies Executor and ParallelExecutor concepts.

#include <dtl/backend/concepts/executor.hpp>

// Include the cpu_executor (adjust path based on actual location)
#include <backends/cpu/cpu_executor.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <vector>

namespace dtl::test {

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(CpuExecutorTest, SatisfiesExecutorConcept) {
    static_assert(Executor<cpu::cpu_executor>,
                  "cpu_executor must satisfy Executor concept");
}

TEST(CpuExecutorTest, SatisfiesParallelExecutorConcept) {
    static_assert(ParallelExecutor<cpu::cpu_executor>,
                  "cpu_executor must satisfy ParallelExecutor concept");
}

TEST(CpuExecutorTest, Name) {
    cpu::cpu_executor exec;
    EXPECT_STREQ(exec.name(), "cpu");
}

// =============================================================================
// Execution Tests
// =============================================================================

TEST(CpuExecutorTest, ExecuteRunsCallable) {
    cpu::cpu_executor exec;

    std::atomic<bool> executed{false};
    exec.execute([&]() { executed = true; });
    exec.synchronize();

    EXPECT_TRUE(executed.load());
}

TEST(CpuExecutorTest, SyncExecuteRunsImmediately) {
    cpu::cpu_executor exec;

    bool executed = false;
    exec.sync_execute([&]() { executed = true; });

    EXPECT_TRUE(executed);
}

TEST(CpuExecutorTest, AsyncExecuteReturnsFuture) {
    cpu::cpu_executor exec;

    auto future = exec.async_execute([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

// =============================================================================
// Parallel For Tests
// =============================================================================

TEST(CpuExecutorTest, ParallelForVisitsAllIndices) {
    cpu::cpu_executor exec;

    constexpr size_type count = 1000;
    std::vector<std::atomic<int>> visited(count);
    for (auto& v : visited) {
        v.store(0);
    }

    exec.parallel_for(count, [&](size_type i) {
        visited[i].fetch_add(1);
    });

    // Verify each index was visited exactly once
    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(visited[i].load(), 1) << "Index " << i << " was not visited exactly once";
    }
}

TEST(CpuExecutorTest, ParallelForWithBeginEnd) {
    cpu::cpu_executor exec;

    constexpr index_t begin = 10;
    constexpr index_t end = 110;
    std::vector<std::atomic<int>> visited(static_cast<size_t>(end));
    for (auto& v : visited) {
        v.store(0);
    }

    exec.parallel_for(begin, end, [&](index_t i) {
        visited[static_cast<size_t>(i)].fetch_add(1);
    });

    // Verify only indices in [begin, end) were visited
    for (index_t i = 0; i < begin; ++i) {
        EXPECT_EQ(visited[static_cast<size_t>(i)].load(), 0)
            << "Index " << i << " should not be visited";
    }
    for (index_t i = begin; i < end; ++i) {
        EXPECT_EQ(visited[static_cast<size_t>(i)].load(), 1)
            << "Index " << i << " was not visited exactly once";
    }
}

TEST(CpuExecutorTest, ParallelForEmptyRange) {
    cpu::cpu_executor exec;

    std::atomic<int> counter{0};
    exec.parallel_for(size_type{0}, [&](size_type) {
        counter.fetch_add(1);
    });

    EXPECT_EQ(counter.load(), 0);
}

// =============================================================================
// Parallel Reduce Tests
// =============================================================================

TEST(CpuExecutorTest, ParallelReduceSum) {
    cpu::cpu_executor exec;

    constexpr index_t count = 1000;

    // Sum of 0..999 = 999*1000/2 = 499500
    int result = exec.parallel_reduce<int>(
        0, count, 0,
        [](index_t begin, index_t end) {
            int sum = 0;
            for (index_t i = begin; i < end; ++i) {
                sum += static_cast<int>(i);
            }
            return sum;
        },
        [](int a, int b) { return a + b; }
    );

    EXPECT_EQ(result, 499500);
}

TEST(CpuExecutorTest, ParallelReduceMax) {
    cpu::cpu_executor exec;

    constexpr index_t count = 1000;
    std::vector<int> data(static_cast<size_t>(count));
    std::iota(data.begin(), data.end(), 0);
    data[500] = 9999;  // Insert maximum

    int result = exec.parallel_reduce<int>(
        0, count, std::numeric_limits<int>::min(),
        [&](index_t begin, index_t end) {
            int max_val = std::numeric_limits<int>::min();
            for (index_t i = begin; i < end; ++i) {
                max_val = std::max(max_val, data[static_cast<size_t>(i)]);
            }
            return max_val;
        },
        [](int a, int b) { return std::max(a, b); }
    );

    EXPECT_EQ(result, 9999);
}

TEST(CpuExecutorTest, ParallelReduceEmptyRange) {
    cpu::cpu_executor exec;

    int result = exec.parallel_reduce<int>(
        0, 0, 42,
        [](index_t, index_t) { return 0; },
        [](int a, int b) { return a + b; }
    );

    EXPECT_EQ(result, 42);  // Should return identity
}

// =============================================================================
// Properties Tests
// =============================================================================

TEST(CpuExecutorTest, MaxParallelismPositive) {
    cpu::cpu_executor exec;
    EXPECT_GT(exec.max_parallelism(), 0);
}

TEST(CpuExecutorTest, SuggestedParallelismPositive) {
    cpu::cpu_executor exec;
    EXPECT_GT(exec.suggested_parallelism(), 0);
}

TEST(CpuExecutorTest, NumThreadsMatchesMaxParallelism) {
    cpu::cpu_executor exec;
    EXPECT_EQ(exec.num_threads(), exec.max_parallelism());
}

TEST(CpuExecutorTest, CustomThreadCount) {
    cpu::cpu_executor exec(4);
    EXPECT_EQ(exec.num_threads(), 4);
    EXPECT_EQ(exec.max_parallelism(), 4);
}

TEST(CpuExecutorTest, Valid) {
    cpu::cpu_executor exec;
    EXPECT_TRUE(exec.valid());
}

// =============================================================================
// Thread Pool Tests
// =============================================================================

TEST(CpuExecutorTest, ThreadPoolSize) {
    cpu::thread_pool pool(4);
    EXPECT_EQ(pool.size(), 4);
}

TEST(CpuExecutorTest, ThreadPoolSubmit) {
    cpu::thread_pool pool(2);

    auto future = pool.submit([]() { return 123; });
    EXPECT_EQ(future.get(), 123);
}

TEST(CpuExecutorTest, ThreadPoolWait) {
    cpu::thread_pool pool(2);

    std::atomic<int> counter{0};
    for (int i = 0; i < 100; ++i) {
        pool.submit([&]() { counter.fetch_add(1); });
    }

    pool.wait();
    EXPECT_EQ(counter.load(), 100);
}

// =============================================================================
// Global Executor Tests
// =============================================================================

TEST(CpuExecutorTest, DefaultExecutorExists) {
    auto& exec = cpu::default_executor();
    EXPECT_TRUE(exec.valid());
}

TEST(CpuExecutorTest, GlobalParallelFor) {
    constexpr size_type count = 100;
    std::vector<std::atomic<int>> visited(count);
    for (auto& v : visited) {
        v.store(0);
    }

    cpu::parallel_for(0, static_cast<index_t>(count), [&](index_t i) {
        visited[static_cast<size_t>(i)].fetch_add(1);
    });

    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(visited[i].load(), 1);
    }
}

TEST(CpuExecutorTest, GlobalParallelReduce) {
    int result = cpu::parallel_reduce<int>(
        0, 100, 0,
        [](index_t begin, index_t end) {
            int sum = 0;
            for (index_t i = begin; i < end; ++i) {
                sum += static_cast<int>(i);
            }
            return sum;
        },
        [](int a, int b) { return a + b; }
    );

    // Sum of 0..99 = 99*100/2 = 4950
    EXPECT_EQ(result, 4950);
}

// =============================================================================
// Move Semantics Tests
// =============================================================================

TEST(CpuExecutorTest, MoveConstructor) {
    cpu::cpu_executor exec1(4);
    cpu::cpu_executor exec2(std::move(exec1));

    EXPECT_TRUE(exec2.valid());
    EXPECT_EQ(exec2.num_threads(), 4);
}

TEST(CpuExecutorTest, MoveAssignment) {
    cpu::cpu_executor exec1(4);
    cpu::cpu_executor exec2(2);

    exec2 = std::move(exec1);

    EXPECT_TRUE(exec2.valid());
    EXPECT_EQ(exec2.num_threads(), 4);
}

}  // namespace dtl::test
