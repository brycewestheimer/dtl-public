// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file futures.hpp
/// @brief Master include for DTL futures support
/// @details Provides single-header access to all futures types.
/// @since 0.1.0
/// @note Updated in 1.3.0: Added background progress, callback executor, diagnostics.

#pragma once

// Distributed future implementation
#include <dtl/futures/distributed_future.hpp>

// Progress engine and completion tracking
#include <dtl/futures/progress.hpp>
#include <dtl/futures/completion.hpp>

// Background progress mode (optional)
#include <dtl/futures/background_progress.hpp>

// Callback executor (isolation)
#include <dtl/futures/callback_executor.hpp>

// Diagnostics and timeout configuration
#include <dtl/futures/diagnostics.hpp>

// Continuation chaining
#include <dtl/futures/continuation.hpp>

// Algorithm result adaptation
#include <dtl/futures/algorithm_result.hpp>

namespace dtl {

// ============================================================================
// Re-export dtl::futures:: types into dtl:: for backward compatibility
// ============================================================================

// distributed_future.hpp types
using futures::shared_state;
using futures::distributed_future;
using futures::distributed_promise;
using futures::make_ready_distributed_future;
using futures::make_failed_distributed_future;

// continuation.hpp types
using futures::continuation_result;
using futures::continuation_result_t;
using futures::when_all;
using futures::when_any;
using futures::on_error;
using futures::chain;
using futures::fmap;
using futures::flatten;
using futures::flat_map;

// algorithm_result.hpp types
using futures::async_policy;
using futures::algorithm_result;
using futures::algorithm_result_t;
using futures::unified_result;
using futures::is_sync_policy_v;
using futures::make_algorithm_result;
using futures::make_algorithm_result_void;
using futures::make_algorithm_error;
using futures::execute_algorithm;

// progress.hpp types (already in dtl::futures::)
using futures::progress_state;
using futures::progress_callback;
using futures::progress_engine;
using futures::poll;
using futures::poll_one;
using futures::poll_for;
using futures::poll_until;
using futures::make_progress;
using futures::drain_progress;
using futures::progress_guard;
using futures::scoped_progress_callback;

// background_progress.hpp types
using futures::progress_mode;
using futures::background_progress_config;
using futures::background_progress_controller;
using futures::start_background_progress;
using futures::stop_background_progress;
using futures::is_background_progress_enabled;
using futures::scoped_background_progress;

// callback_executor.hpp types
using futures::executor_mode;
using futures::executor_config;
using futures::callback_executor;
using futures::global_callback_executor;

// diagnostics.hpp types
using futures::timeout_config;
using futures::pending_future_info;
using futures::progress_diagnostics;
using futures::diagnostic_collector;
using futures::timeout_exception;
using futures::global_timeout_config;
using futures::set_global_timeout_config;
using futures::effective_wait_timeout;

// completion.hpp types (already in dtl::futures::)
using futures::completion_token;
using futures::completion_set;
using futures::completion_waiter;

// ============================================================================
// Futures Module Summary
// ============================================================================
//
// The futures module provides abstractions for asynchronous distributed
// computations. It integrates with DTL's execution policies to provide
// both synchronous and asynchronous algorithm execution.
//
// ============================================================================
// Distributed Future
// ============================================================================
//
// distributed_future<T> represents an asynchronous computation:
//
// - get(): Wait and return value (throws on error)
// - get_result(): Wait and return result<T> (non-throwing)
// - wait(): Block until ready
// - wait_for(duration): Wait with timeout
// - is_ready(): Non-blocking check
// - valid(): Check if future has associated state
// - then(func): Add continuation
//
// Special case:
// - distributed_future<void> for operations without return value
//
// ============================================================================
// Promise
// ============================================================================
//
// distributed_promise<T> sets the value for a future:
//
// - get_future(): Get the associated future
// - set_value(T): Set the result value
// - set_error(status): Set an error
//
// ============================================================================
// Factory Functions
// ============================================================================
//
// Creating futures:
// - make_ready_distributed_future(value): Create ready future
// - make_ready_distributed_future(): Create ready void future
// - make_failed_distributed_future<T>(error): Create failed future
//
// ============================================================================
// Continuations
// ============================================================================
//
// Chain computations with .then():
//
// @code
// auto future = compute_async()
//     .then([](int x) { return x * 2; })
//     .then([](int x) { return std::to_string(x); });
// @endcode
//
// Combinators:
// - when_all(futures...): Wait for all futures
// - when_any(futures): Wait for first to complete
// - chain(future, funcs...): Chain multiple continuations
// - fmap(future, func): Map function over future
// - flatten(nested): Flatten nested future
// - flat_map(future, func): Monadic bind
//
// ============================================================================
// Algorithm Results
// ============================================================================
//
// Return types adapt based on execution policy:
//
// - seq{}/par{}: Returns result<T> (synchronous)
// - async{}: Returns distributed_future<T> (asynchronous)
//
// Type traits:
// - algorithm_result_t<Policy, T>: Get return type for policy
// - is_seq_policy_v<Policy>: Check if sequential
// - is_par_policy_v<Policy>: Check if parallel
// - is_async_policy_v<Policy>: Check if asynchronous
// - is_sync_policy_v<Policy>: Check if synchronous (seq or par)
//
// Helper functions:
// - make_algorithm_result<Policy>(value): Create result
// - make_algorithm_result_void<Policy>(): Create void result
// - make_algorithm_error<Policy, T>(error): Create error result
// - execute_algorithm<Policy>(func): Execute with appropriate return type
//
// ============================================================================
// Unified Result
// ============================================================================
//
// unified_result<T> holds either sync or async result:
//
// - is_async(): Check if async
// - get(): Get as result<T> (blocks if async)
// - get_future(): Get as future (creates ready future if sync)
// - wait(): Wait for completion
// - is_ready(): Non-blocking ready check
//
// ============================================================================
// Usage Examples
// ============================================================================
//
// @code
// #include <dtl/futures/futures.hpp>
//
// // Create and fulfill a promise
// dtl::distributed_promise<int> promise;
// auto future = promise.get_future();
//
// std::thread([&promise] {
//     promise.set_value(42);
// }).detach();
//
// int value = future.get();  // Blocks until value is set
//
// // Chain continuations
// auto result = compute()
//     .then([](int x) { return x + 1; })
//     .then([](int x) { return x * 2; })
//     .get();
//
// // Wait for multiple futures
// auto combined = dtl::when_all(future1, future2, future3);
// auto [v1, v2, v3] = combined.get();
//
// // Algorithm with execution policy
// auto sum = dtl::reduce(dtl::async{}, vec, 0, std::plus<>{});
// // sum is distributed_future<int>
//
// auto sum2 = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{});
// // sum2 is result<int>
// @endcode
//
// ============================================================================
// Integration with Algorithms
// ============================================================================
//
// Algorithms automatically adapt their return type based on execution policy:
//
// @code
// template <typename Policy, typename Container, typename T, typename Op>
// algorithm_result_t<Policy, T> reduce(Policy, Container& c, T init, Op op) {
//     if constexpr (is_async_policy_v<Policy>) {
//         // Return future
//         distributed_promise<T> promise;
//         auto future = promise.get_future();
//         // ... launch async work ...
//         return future;
//     } else {
//         // Return result
//         T result = init;
//         // ... compute synchronously ...
//         return result;
//     }
// }
// @endcode
//
// ============================================================================

}  // namespace dtl
