// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file predicates.hpp
/// @brief Distributed predicate algorithms
/// @details all_of, any_of, none_of for distributed containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <memory>

namespace dtl {

// ============================================================================
// Distributed all_of
// ============================================================================

/// @brief Check if predicate is true for all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return true if predicate is true for all elements globally
///
/// @par Complexity:
/// O(n/p) local evaluations, plus O(log p) allreduce communication.
/// Short-circuits locally but must communicate to get global result.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// bool all_positive = dtl::all_of(dtl::par{}, vec,
///     [](int x) { return x > 0; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool all_of([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::all_of");

    // Check locally with short-circuit
    bool local_result = true;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (!pred(*it)) {
            local_result = false;
            break;
        }
    }

    // Standalone mode: return local result only
    // For distributed all_of, use overload with communicator parameter
    return local_result;
}

/// @brief Check if predicate is true for all elements (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("all_of(container, pred) is local-only for multi-rank containers; use all_of(..., comm) for collective semantics or local_all_of(...) for rank-local semantics")
bool all_of(const Container& container, Predicate pred) {
    return all_of(seq{}, container, std::move(pred));
}

/// @brief Check if predicate is true for all elements with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @param comm The communicator for allreduce
/// @return true if predicate is true for all elements globally
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// Uses allreduce with logical AND to combine results.
template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
bool all_of([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    // Check locally with short-circuit
    bool local_result = true;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (!pred(*it)) {
            local_result = false;
            break;
        }
    }

    // Allreduce with logical AND
    return comm.allreduce_land_value(local_result);
}

/// @brief Check if predicate is true for all elements with communicator (default execution)
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
bool all_of(const Container& container, Predicate pred, Comm& comm) {
    return all_of(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Distributed any_of
// ============================================================================

/// @brief Check if predicate is true for any element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return true if predicate is true for at least one element globally
///
/// @par Complexity:
/// O(n/p) local evaluations, plus O(log p) allreduce communication.
///
/// @par Example:
/// @code
/// bool has_negative = dtl::any_of(dtl::par{}, vec,
///     [](int x) { return x < 0; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool any_of([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::any_of");

    // Check locally with short-circuit
    bool local_result = false;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            local_result = true;
            break;
        }
    }

    // Standalone mode: return local result only
    // For distributed any_of, use overload with communicator parameter
    return local_result;
}

/// @brief Check if predicate is true for any element (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("any_of(container, pred) is local-only for multi-rank containers; use any_of(..., comm) for collective semantics or local_any_of(...) for rank-local semantics")
bool any_of(const Container& container, Predicate pred) {
    return any_of(seq{}, container, std::move(pred));
}

/// @brief Check if predicate is true for any element with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @param comm The communicator for allreduce
/// @return true if predicate is true for at least one element globally
template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
bool any_of([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    // Check locally with short-circuit
    bool local_result = false;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            local_result = true;
            break;
        }
    }

    // Allreduce with logical OR
    return comm.allreduce_lor_value(local_result);
}

/// @brief Check if predicate is true for any element with communicator (default execution)
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
bool any_of(const Container& container, Predicate pred, Comm& comm) {
    return any_of(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Distributed none_of
// ============================================================================

/// @brief Check if predicate is false for all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return true if predicate is false for all elements globally
///
/// @par Equivalence:
/// Equivalent to `!any_of(container, pred)`.
///
/// @par Example:
/// @code
/// bool no_zeros = dtl::none_of(dtl::par{}, vec,
///     [](int x) { return x == 0; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool none_of(ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::none_of");
    return !any_of(std::forward<ExecutionPolicy>(policy), container, std::move(pred));
}

/// @brief Check if predicate is false for all elements (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("none_of(container, pred) is local-only for multi-rank containers; use none_of(..., comm) for collective semantics or local_none_of(...) for rank-local semantics")
bool none_of(const Container& container, Predicate pred) {
    return none_of(seq{}, container, std::move(pred));
}

/// @brief Check if predicate is false for all elements with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @param comm The communicator for allreduce
/// @return true if predicate is false for all elements globally
template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
bool none_of(ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    return !any_of(std::forward<ExecutionPolicy>(policy), container, std::move(pred), comm);
}

/// @brief Check if predicate is false for all elements with communicator (default execution)
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
bool none_of(const Container& container, Predicate pred, Comm& comm) {
    return none_of(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Local-only predicates (no communication)
// ============================================================================

/// @brief Check if predicate is true for all local elements (no communication)
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param container The distributed container
/// @param pred Unary predicate
/// @return true if predicate is true for all local elements
///
/// @note NOT collective - checks local partition only.
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
bool local_all_of(const Container& container, Predicate pred) {
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (!pred(*it)) {
            return false;
        }
    }
    return true;
}

/// @brief Check if predicate is true for any local element (no communication)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
bool local_any_of(const Container& container, Predicate pred) {
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            return true;
        }
    }
    return false;
}

/// @brief Check if predicate is false for all local elements (no communication)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
bool local_none_of(const Container& container, Predicate pred) {
    return !local_any_of(container, std::move(pred));
}

// ============================================================================
// Async predicates
// ============================================================================

/// @brief Asynchronously check all_of
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Future containing boolean result
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto async_all_of(const Container& container, Predicate pred)
    -> futures::distributed_future<bool> {
    auto promise = std::make_shared<futures::distributed_promise<bool>>();
    auto future = promise->get_future();

    try {
        auto result = all_of(seq{}, container, std::move(pred));
        promise->set_value(result);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously check any_of
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Future containing boolean result
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto async_any_of(const Container& container, Predicate pred)
    -> futures::distributed_future<bool> {
    auto promise = std::make_shared<futures::distributed_promise<bool>>();
    auto future = promise->get_future();

    try {
        auto result = any_of(seq{}, container, std::move(pred));
        promise->set_value(result);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously check none_of
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Future containing boolean result
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto async_none_of(const Container& container, Predicate pred)
    -> futures::distributed_future<bool> {
    auto promise = std::make_shared<futures::distributed_promise<bool>>();
    auto future = promise->get_future();

    try {
        auto result = none_of(seq{}, container, std::move(pred));
        promise->set_value(result);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
