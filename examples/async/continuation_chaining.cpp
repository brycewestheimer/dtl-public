// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file continuation_chaining.cpp
/// @brief Future continuation patterns using .then() chaining
/// @details Demonstrates DTL's distributed_future with continuation chaining,
///          when_all, when_any, and error handling patterns.
///
/// Build:
///   mkdir build && cd build
///   cmake .. -DDTL_BUILD_EXAMPLES=ON
///   make
///
/// Run:
///   ./continuation_chaining
///
/// Expected output:
///   DTL Continuation Chaining Example
///   =================================
///
///   --- Example 1: Simple .then() Chain ---
///   Initial value: 10
///   After doubling: 20
///   After adding 5: 25
///   After converting to string: "Result: 25"
///
///   --- Example 2: when_all - Wait for Multiple Futures ---
///   Starting 3 parallel computations...
///   All computations complete.
///   Results: 100, 200, 300
///   Sum: 600
///
///   --- Example 3: Computation Graph ---
///   Building: result = (a + b) * (c - d)
///   a=10, b=20, c=50, d=15
///   (a + b) = 30
///   (c - d) = 35
///   result = 1050
///
///   SUCCESS: Continuation chaining example completed!

#include <dtl/dtl.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

// Simulate async computation with a delay
template <typename T>
dtl::distributed_future<T> async_compute(T value, int delay_ms = 10) {
    dtl::distributed_promise<T> promise;
    auto future = promise.get_future();

    // In a real implementation, this would be truly async
    // For this example, we compute immediately
    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    promise.set_value(value);

    return future;
}

int main() {
    std::cout << "DTL Continuation Chaining Example\n";
    std::cout << "=================================\n\n";

    // =========================================================================
    // Example 1: Simple .then() Chain
    // =========================================================================
    std::cout << "--- Example 1: Simple .then() Chain ---\n";

    int initial = 10;
    std::cout << "Initial value: " << initial << "\n";

    // Create initial future
    auto future1 = dtl::make_ready_distributed_future(initial);

    // Chain transformations
    auto doubled = future1.then([](int x) {
        std::cout << "After doubling: " << (x * 2) << "\n";
        return x * 2;
    });

    auto plus_five = doubled.then([](int x) {
        std::cout << "After adding 5: " << (x + 5) << "\n";
        return x + 5;
    });

    auto as_string = plus_five.then([](int x) {
        std::string result = "Result: " + std::to_string(x);
        std::cout << "After converting to string: \"" << result << "\"\n";
        return result;
    });

    // Wait for final result
    std::string final_string = as_string.get();

    std::cout << "\n";

    // =========================================================================
    // Example 2: when_all - Wait for Multiple Futures
    // =========================================================================
    std::cout << "--- Example 2: when_all - Wait for Multiple Futures ---\n";
    std::cout << "Starting 3 parallel computations...\n";

    auto f1 = async_compute(100, 5);
    auto f2 = async_compute(200, 10);
    auto f3 = async_compute(300, 3);

    // Wait for all futures
    auto all_done = dtl::when_all(std::move(f1), std::move(f2), std::move(f3));

    // Get results - returns tuple
    auto [r1, r2, r3] = all_done.get();

    std::cout << "All computations complete.\n";
    std::cout << "Results: " << r1 << ", " << r2 << ", " << r3 << "\n";
    std::cout << "Sum: " << (r1 + r2 + r3) << "\n\n";

    // =========================================================================
    // Example 3: Computation Graph
    // =========================================================================
    std::cout << "--- Example 3: Computation Graph ---\n";
    std::cout << "Building: result = (a + b) * (c - d)\n";

    int a = 10, b = 20, c = 50, d = 15;
    std::cout << "a=" << a << ", b=" << b << ", c=" << c << ", d=" << d << "\n";

    // Create leaf futures
    auto fa = dtl::make_ready_distributed_future(a);
    auto fb = dtl::make_ready_distributed_future(b);
    auto fc = dtl::make_ready_distributed_future(c);
    auto fd = dtl::make_ready_distributed_future(d);

    // First level: (a + b) and (c - d)
    auto fab_future = dtl::when_all(std::move(fa), std::move(fb));
    auto fcd_future = dtl::when_all(std::move(fc), std::move(fd));

    auto sum_ab = fab_future.then([](std::tuple<int, int> vals) {
        auto [va, vb] = vals;
        int result = va + vb;
        std::cout << "(a + b) = " << result << "\n";
        return result;
    });

    auto diff_cd = fcd_future.then([](std::tuple<int, int> vals) {
        auto [vc, vd] = vals;
        int result = vc - vd;
        std::cout << "(c - d) = " << result << "\n";
        return result;
    });

    // Second level: multiply results
    auto final_calc = dtl::when_all(std::move(sum_ab), std::move(diff_cd));
    auto result = final_calc.then([](std::tuple<int, int> vals) {
        auto [sum, diff] = vals;
        return sum * diff;
    });

    int final_result = result.get();
    std::cout << "result = " << final_result << "\n\n";

    // Verify
    int expected = (a + b) * (c - d);
    if (final_result == expected) {
        std::cout << "SUCCESS: Continuation chaining example completed!\n";
        return 0;
    } else {
        std::cout << "FAILURE: Expected " << expected << " but got " << final_result << "\n";
        return 1;
    }
}
