// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file for_each.hpp
/// @brief Distributed for_each algorithm
/// @details Apply function to each element in a distributed range.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <functional>
#include <memory>

namespace dtl {

// ============================================================================
// Distributed for_each
// ============================================================================

/// @brief Apply function to each element in a distributed range
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam UnaryFunction Function type
/// @param policy Execution policy controlling parallelism
/// @param container The distributed container
/// @param f Function to apply to each element
/// @return The function object after processing
///
/// @par Complexity:
/// O(n/p) local operations where n is global size and p is number of ranks.
/// No communication occurs - operates on local partition only.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// // Sequential local processing
/// dtl::for_each(dtl::seq{}, vec, [](int& x) { x *= 2; });
///
/// // Parallel local processing
/// dtl::for_each(dtl::par{}, vec, [](int& x) { x *= 2; });
/// @endcode
///
/// @note Operates on local partition only.
///       For distributed iteration, use segmented_for_each.
template <typename ExecutionPolicy, typename Container, typename UnaryFunction>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
UnaryFunction for_each(ExecutionPolicy&& policy,
                       Container& container,
                       UnaryFunction f) {
    auto local_v = container.local_view();
    dispatch_for_each(std::forward<ExecutionPolicy>(policy),
                      local_v.begin(), local_v.end(), f);
    return f;
}

/// @brief Apply function to each element (default sequential execution)
/// @tparam Container Distributed container type
/// @tparam UnaryFunction Function type
/// @param container The distributed container
/// @param f Function to apply
/// @return The function object
template <typename Container, typename UnaryFunction>
    requires DistributedContainer<Container>
UnaryFunction for_each(Container& container, UnaryFunction f) {
    return for_each(seq{}, container, std::move(f));
}

// ============================================================================
// Distributed for_each_n
// ============================================================================

/// @brief Apply function to first n elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Size Count type
/// @tparam UnaryFunction Function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param n Number of elements to process (local count)
/// @param f Function to apply
/// @return Iterator past the last processed element
///
/// @note n is the count of local elements to process from this rank's partition.
template <typename ExecutionPolicy, typename Container, typename Size, typename UnaryFunction>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             std::integral<Size>
auto for_each_n(ExecutionPolicy&& policy,
                Container& container,
                Size n,
                UnaryFunction f) {
    auto local_v = container.local_view();
    Size actual_n = std::min(static_cast<Size>(local_v.size()), n);
    dispatch_for_each(std::forward<ExecutionPolicy>(policy),
                      local_v.begin(), local_v.begin() + actual_n, f);
    return local_v.begin() + actual_n;
}

// ============================================================================
// Segmented for_each (Primary Distributed Pattern)
// ============================================================================

/// @brief Apply function using segmented iteration
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam UnaryFunction Function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param f Function to apply
/// @return Result indicating success or failure
///
/// @par Design Rationale:
/// This is the preferred pattern for distributed iteration.
/// Uses segmented_view to process each rank's local data efficiently.
///
/// @par Three-Phase Pattern:
/// 1. Local computation via segmented iteration
/// 2. Synchronization at phase boundary (barrier)
/// 3. Collective operations if needed
///
/// @par Example:
/// @code
/// dtl::segmented_for_each(dtl::par{}, vec, [](int& x) { x *= 2; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename UnaryFunction>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> segmented_for_each(ExecutionPolicy&& policy,
                                Container& container,
                                UnaryFunction f) {
    // Phase 1: Local computation using segmented iteration
    auto segmented = container.segmented_view();
    for (auto segment : segmented) {
        if (segment.is_local()) {
            // Use dispatch to handle execution policy
            dispatch_for_each(std::forward<ExecutionPolicy>(policy),
                              segment.begin(), segment.end(), f);
        }
    }
    // Phase 2: No barrier needed for for_each (no communication)
    // Phase 3: No collective operations needed
    return {};
}

// ============================================================================
// Async for_each
// ============================================================================

/// @brief Asynchronously apply function to each element
/// @tparam Container Distributed container type
/// @tparam UnaryFunction Function type
/// @param container The distributed container
/// @param f Function to apply
/// @return Future that completes when processing is done
///
/// @note Returns immediately; use .get() or .then() to await completion.
template <typename Container, typename UnaryFunction>
    requires DistributedContainer<Container>
auto async_for_each(Container& container, UnaryFunction f)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        for_each(seq{}, container, std::move(f));
        promise->set_value();
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
