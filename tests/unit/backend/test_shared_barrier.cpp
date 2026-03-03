// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shared_barrier.cpp
/// @brief Tests for shared_memory::shared_barrier (C++20 atomic wait)

#include <gtest/gtest.h>

// Suppress deprecation warning for test inclusion
#define DTL_ENABLE_EXPERIMENTAL_BACKENDS
#include <backends/shared_memory/shared_memory_communicator.hpp>

#include <atomic>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Single-thread barrier (count=1): must not deadlock
// ---------------------------------------------------------------------------
TEST(SharedBarrierTest, SingleThread) {
    dtl::shared_memory::shared_barrier barrier(1);
    barrier.arrive_and_wait();
    barrier.arrive_and_wait();  // reuse
}

// ---------------------------------------------------------------------------
// Two threads synchronize correctly
// ---------------------------------------------------------------------------
TEST(SharedBarrierTest, TwoThreads) {
    dtl::shared_memory::shared_barrier barrier(2);
    std::atomic<int> counter{0};

    std::thread t([&] {
        counter.fetch_add(1, std::memory_order_relaxed);
        barrier.arrive_and_wait();
        // After barrier, both threads have incremented
        EXPECT_EQ(counter.load(std::memory_order_relaxed), 2);
    });

    counter.fetch_add(1, std::memory_order_relaxed);
    barrier.arrive_and_wait();
    EXPECT_EQ(counter.load(std::memory_order_relaxed), 2);

    t.join();
}

// ---------------------------------------------------------------------------
// Multiple threads synchronize correctly
// ---------------------------------------------------------------------------
TEST(SharedBarrierTest, MultipleThreads) {
    constexpr int num_threads = 8;
    dtl::shared_memory::shared_barrier barrier(num_threads);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&] {
            counter.fetch_add(1, std::memory_order_relaxed);
            barrier.arrive_and_wait();
            EXPECT_EQ(counter.load(std::memory_order_relaxed), num_threads);
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

// ---------------------------------------------------------------------------
// Repeated barrier usage across multiple rounds
// ---------------------------------------------------------------------------
TEST(SharedBarrierTest, RepeatedBarrier) {
    constexpr int num_threads = 4;
    constexpr int num_rounds = 10;
    dtl::shared_memory::shared_barrier barrier(num_threads);
    std::atomic<int> counter{0};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&] {
            for (int round = 0; round < num_rounds; ++round) {
                counter.fetch_add(1, std::memory_order_relaxed);
                barrier.arrive_and_wait();
                int expected = (round + 1) * num_threads;
                EXPECT_EQ(counter.load(std::memory_order_relaxed), expected);
                barrier.arrive_and_wait();  // sync before next round
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(std::memory_order_relaxed),
              num_threads * num_rounds);
}

// ---------------------------------------------------------------------------
// Barrier with work between phases
// ---------------------------------------------------------------------------
TEST(SharedBarrierTest, PhasedWork) {
    constexpr int num_threads = 4;
    dtl::shared_memory::shared_barrier barrier(num_threads);
    std::vector<int> results(num_threads, 0);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i] {
            // Phase 1: each thread writes its own slot
            results[static_cast<size_t>(i)] = i + 1;
            barrier.arrive_and_wait();

            // Phase 2: verify all slots are written
            for (int j = 0; j < num_threads; ++j) {
                EXPECT_EQ(results[static_cast<size_t>(j)], j + 1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

}  // namespace
