// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file iota.hpp
/// @brief Distributed iota algorithm (standalone header)
/// @details Fill distributed containers with sequentially increasing values,
///          with proper global offset awareness for multi-rank environments.
///          This is the dedicated iota header; the fill.hpp also contains
///          basic iota/global_iota for backward compatibility.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <memory>
#include <numeric>

namespace dtl {

// ============================================================================
// Distributed Iota (global offset-aware)
// ============================================================================

/// @brief Fill with globally-correct increasing sequence
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type (must support addition with index_t)
/// @param policy Execution policy
/// @param container Container to fill
/// @param start Starting value at global index 0
/// @return Result indicating success or failure
///
/// @par Behavior:
/// Each element at global index `g` gets value: start + g
/// Internally, each rank computes its local starting value as:
///   local_start = start + global_offset_of_this_rank
/// Then fills local elements with local_start, local_start+1, ...
///
/// @par Complexity:
/// O(n/p) local assignments. No communication required.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// dtl::distributed_iota(dtl::par{}, vec, 0);
/// // rank 0: [0, 1, ..., 249], rank 1: [250, 251, ..., 499], etc.
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> distributed_iota([[maybe_unused]] ExecutionPolicy&& policy,
                               Container& container,
                               T start) {
    auto local_v = container.local_view();
    T local_start = start + static_cast<T>(local_v.global_offset());
    std::iota(local_v.begin(), local_v.end(), local_start);
    return {};
}

/// @brief Distributed iota with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
result<void> distributed_iota(Container& container, T start) {
    return distributed_iota(seq{}, container, start);
}

// ============================================================================
// Local Iota (no offset awareness)
// ============================================================================

/// @brief Fill local partition with increasing sequence starting from value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param start Starting value (local, not adjusted for global offset)
/// @return Number of elements filled
///
/// @note NOT globally aware - fills local data starting from `start` regardless
///       of the container's position in the global distribution.
///       Use distributed_iota() for globally-correct sequences.
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type local_iota([[maybe_unused]] ExecutionPolicy&& policy,
                     Container& container,
                     T start) {
    auto local_v = container.local_view();
    std::iota(local_v.begin(), local_v.end(), start);
    return local_v.size();
}

/// @brief Local iota with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_iota(Container& container, T start) {
    return local_iota(seq{}, container, start);
}

// ============================================================================
// Iota with Custom Step
// ============================================================================

/// @brief Fill with sequence using a custom step value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to fill
/// @param start Starting value at global index 0
/// @param step Increment between consecutive elements
/// @return Result indicating success or failure
///
/// @par Behavior:
/// Each element at global index `g` gets value: start + g * step
///
/// @par Example:
/// @code
/// distributed_vector<double> vec(100, ctx);
/// dtl::iota_step(dtl::seq{}, vec, 0.0, 0.5);
/// // vec = [0.0, 0.5, 1.0, 1.5, ...]
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> iota_step([[maybe_unused]] ExecutionPolicy&& policy,
                       Container& container,
                       T start,
                       T step) {
    auto local_v = container.local_view();
    T base = start + static_cast<T>(local_v.global_offset()) * step;
    for (size_type i = 0; i < local_v.size(); ++i) {
        local_v[i] = base + static_cast<T>(i) * step;
    }
    return {};
}

/// @brief Iota with step and default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
result<void> iota_step(Container& container, T start, T step) {
    return iota_step(seq{}, container, start, step);
}

// ============================================================================
// Async Iota
// ============================================================================

/// @brief Asynchronously fill with increasing sequence
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container Container to fill
/// @param start Starting value
/// @return Future indicating completion
template <typename Container, typename T>
    requires DistributedContainer<Container>
auto async_distributed_iota(Container& container, T start)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto res = distributed_iota(seq{}, container, start);
        if (res) {
            promise->set_value();
        } else {
            promise->set_error(res.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
