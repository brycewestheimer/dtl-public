// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file fill.hpp
/// @brief Distributed fill algorithm
/// @details Set all elements to a specified value.
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

#include <algorithm>
#include <memory>
#include <numeric>

namespace dtl {

// ============================================================================
// Distributed fill
// ============================================================================

/// @brief Fill all elements with a value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param value Value to assign to all elements
/// @return Result indicating success or failure
///
/// @par Complexity:
/// O(n/p) local assignments where n is global size and p is ranks.
/// No communication required (embarrassingly parallel).
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// dtl::fill(dtl::par{}, vec, 42);  // All elements = 42
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> fill(ExecutionPolicy&& policy, Container& container, const T& value) {
    auto local_v = container.local_view();
    dispatch_fill(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.end(), value);
    return {};
}

/// @brief Fill with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
result<void> fill(Container& container, const T& value) {
    return fill(seq{}, container, value);
}

// ============================================================================
// Fill_n
// ============================================================================

/// @brief Fill first n elements with a value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param n Number of elements to fill (local count)
/// @param value Value to assign
/// @return Number of elements filled in local partition
///
/// @note n is the number of local elements to fill from this rank's partition.
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type fill_n(ExecutionPolicy&& policy,
                 Container& container,
                 size_type n,
                 const T& value) {
    auto local_v = container.local_view();
    size_type actual_n = std::min(n, local_v.size());
    dispatch_fill(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.begin() + actual_n, value);
    return actual_n;
}

/// @brief Fill_n with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type fill_n(Container& container, size_type n, const T& value) {
    return fill_n(seq{}, container, n, value);
}

// ============================================================================
// Generate
// ============================================================================

/// @brief Fill elements using a generator function
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Generator Generator function type
/// @param policy Execution policy
/// @param container Container to fill
/// @param gen Generator function (called once per element)
/// @return Result indicating success or failure
///
/// @warning Generator is called independently per rank. For consistent
///          global sequences, use iota or explicit indexing.
///
/// @par Example:
/// @code
/// int counter = 0;
/// dtl::generate(dtl::seq{}, vec, [&counter]() { return counter++; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Generator>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> generate(ExecutionPolicy&& policy,
                      Container& container,
                      Generator gen) {
    auto local_v = container.local_view();
    // Note: generate cannot be easily parallelized if gen has state
    std::generate(local_v.begin(), local_v.end(), std::move(gen));
    return {};
}

/// @brief Generate with default execution
template <typename Container, typename Generator>
    requires DistributedContainer<Container>
result<void> generate(Container& container, Generator gen) {
    return generate(seq{}, container, std::move(gen));
}

// ============================================================================
// Iota
// ============================================================================

/// @brief Fill with increasing sequence starting from value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param start Starting value
/// @return Result indicating success or failure
///
/// @par Global vs Local Behavior:
/// In single-rank mode, elements are assigned start, start+1, start+2, ...
/// In multi-rank mode (with Phase 4 backend), elements are assigned their
/// global index + start value, ensuring a consistent global sequence.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// dtl::iota(dtl::par{}, vec, 0);  // vec[i] = i globally
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> iota(ExecutionPolicy&& policy,
                  Container& container,
                  T start) {
    auto local_v = container.local_view();
    // In full implementation, would adjust start by global offset
    std::iota(local_v.begin(), local_v.end(), start);
    return {};
}

/// @brief Iota with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
result<void> iota(Container& container, T start) {
    return iota(seq{}, container, start);
}

/// @brief Iota with global offset awareness
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container Container to fill
/// @param start Starting value at global index 0
/// @return Result indicating success or failure
///
/// @par Behavior:
/// Each element gets value: start + global_index
template <typename Container, typename T>
    requires DistributedContainer<Container>
result<void> global_iota(Container& container, T start) {
    auto local_v = container.local_view();
    // Use global_offset to compute correct starting value for this rank
    T local_start = start + static_cast<T>(local_v.global_offset());
    std::iota(local_v.begin(), local_v.end(), local_start);
    return {};
}

// ============================================================================
// Local-only fill (no communication)
// ============================================================================

/// @brief Fill local partition only
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param value Value to assign
/// @return Number of elements filled
///
/// @note NOT collective - fills local data only.
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type local_fill(ExecutionPolicy&& policy, Container& container, const T& value) {
    auto local_v = container.local_view();
    dispatch_fill(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.end(), value);
    return local_v.size();
}

/// @brief Local fill with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_fill(Container& container, const T& value) {
    return local_fill(seq{}, container, value);
}

// ============================================================================
// Async fill
// ============================================================================

/// @brief Asynchronously fill elements
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container Container to fill
/// @param value Value to assign
/// @return Future indicating completion
template <typename Container, typename T>
    requires DistributedContainer<Container>
auto async_fill(Container& container, const T& value)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = fill(seq{}, container, value);
        if (result) {
            promise->set_value();
        } else {
            promise->set_error(result.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
