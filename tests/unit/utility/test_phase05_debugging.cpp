// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_phase05_debugging.cpp
/// @brief Phase 05 tests: debugging.hpp fixes
/// @details Tests for T04 (timed scope macro), T05 (global state thread safety),
///          and T06 (chrono include).

#include <dtl/utility/debugging.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>
#include <vector>

namespace dtl::test {

// =============================================================================
// T06: <chrono> include — the fact that this file compiles with debugging.hpp
// as the first DTL include verifies the include is self-contained.
// =============================================================================

TEST(Phase05DebuggingTest, ChronoIncludePresent) {
    // If this compiles, <chrono> is properly included
    auto now = std::chrono::high_resolution_clock::now();
    EXPECT_GT(now.time_since_epoch().count(), 0);
}

// =============================================================================
// T04: DTL_TIMED_SCOPE macro properly expands __LINE__
// =============================================================================

TEST(Phase05DebuggingTest, TimedScopeCompilesTwice) {
    // Redirect debug output to suppress timer output during test
    std::ostringstream oss;
    dtl::set_debug_stream(oss);
    auto prev_level = dtl::get_debug_level();
    dtl::set_debug_level(dtl::debug_level::info);

    {
        DTL_TIMED_SCOPE("first_timer");
        DTL_TIMED_SCOPE("second_timer");
        // If __LINE__ expansion is broken, both would have the same variable name
        // and this would fail to compile.
    }

    // Restore
    dtl::set_debug_level(prev_level);
    dtl::set_debug_stream(std::cerr);
}

TEST(Phase05DebuggingTest, TimedScopeRecordsElapsed) {
    std::ostringstream oss;
    dtl::set_debug_stream(oss);
    dtl::set_debug_level(dtl::debug_level::info);

    {
        dtl::scoped_timer timer("test_elapsed", dtl::debug_level::info);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        EXPECT_GE(timer.elapsed(), 0.005); // at least 5ms
    }

    // Timer destructor should have written output
    std::string output = oss.str();
    EXPECT_NE(output.find("test_elapsed"), std::string::npos);
    EXPECT_NE(output.find("took"), std::string::npos);

    dtl::set_debug_stream(std::cerr);
}

// =============================================================================
// T05: Global state thread safety
// =============================================================================

TEST(Phase05DebuggingTest, ConcurrentDebugLevelAccess) {
    constexpr int num_threads = 8;
    constexpr int iterations = 1000;
    std::atomic<bool> start{false};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load()) {}  // spin until all threads ready

            for (int i = 0; i < iterations; ++i) {
                if (t % 2 == 0) {
                    // Writers
                    dtl::set_debug_level(
                        static_cast<dtl::debug_level>(i % 6));
                } else {
                    // Readers
                    auto level = dtl::get_debug_level();
                    EXPECT_GE(static_cast<int>(level), 0);
                    EXPECT_LE(static_cast<int>(level), 5);
                }
            }
        });
    }

    start.store(true);

    for (auto& t : threads) {
        t.join();
    }
}

TEST(Phase05DebuggingTest, ConcurrentDebugRankAccess) {
    constexpr int num_threads = 4;
    constexpr int iterations = 1000;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < iterations; ++i) {
                dtl::set_debug_rank(t);
                // No crash or data race
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

TEST(Phase05DebuggingTest, ConcurrentConfigureDebug) {
    constexpr int num_threads = 4;
    constexpr int iterations = 500;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < iterations; ++i) {
                dtl::configure_debug(t % 2 == 0, t % 3 == 0);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

TEST(Phase05DebuggingTest, DebugLevelSetAndGet) {
    dtl::set_debug_level(dtl::debug_level::trace);
    EXPECT_EQ(dtl::get_debug_level(), dtl::debug_level::trace);

    dtl::set_debug_level(dtl::debug_level::error);
    EXPECT_EQ(dtl::get_debug_level(), dtl::debug_level::error);

    dtl::set_debug_level(dtl::debug_level::none);
    EXPECT_EQ(dtl::get_debug_level(), dtl::debug_level::none);
}

TEST(Phase05DebuggingTest, ConcurrentDebugOutput) {
    // Redirect output to suppress noise
    std::ostringstream oss;
    dtl::set_debug_stream(oss);
    dtl::set_debug_level(dtl::debug_level::trace);

    constexpr int num_threads = 4;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < 50; ++i) {
                dtl::detail::debug_output(
                    dtl::debug_level::info,
                    std::source_location::current(),
                    "thread ", t, " iteration ", i);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Restore
    dtl::set_debug_stream(std::cerr);
    dtl::set_debug_level(dtl::debug_level::warning);
}

}  // namespace dtl::test
