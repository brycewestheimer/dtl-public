// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file find.hpp
/// @brief Distributed find algorithms
/// @details Locate elements in distributed containers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/detail/determinism_guard.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <optional>
#include <limits>
#include <stdexcept>

namespace dtl {

// ============================================================================
// Find Result Type
// ============================================================================

/// @brief Result of a distributed find operation
/// @tparam T Element type
template <typename T>
struct find_result {
    /// @brief Whether an element was found
    bool found = false;

    /// @brief Global index of found element (valid only if found)
    index_t global_index = 0;

    /// @brief Rank that found the element (valid only if found)
    rank_t owner_rank = no_rank;

    /// @brief The found value (valid only if found and requested)
    std::optional<T> value = std::nullopt;

    /// @brief Check if element was found
    explicit operator bool() const noexcept { return found; }
};

// ============================================================================
// Distributed find
// ============================================================================

/// @brief Find first element equal to value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container The distributed container
/// @param value Value to search for
/// @return Result containing find information
///
/// @par Complexity:
/// O(n/p) local comparisons, plus O(log p) communication for reduction.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result is consistent across all ranks.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto result = dtl::find(dtl::par{}, vec, 42);
/// if (result) {
///     std::cout << "Found at global index " << result.global_index << "\n";
/// }
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
find_result<typename Container::value_type>
find([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, const T& value) {
    if (container.num_ranks() > 1) {
        throw std::runtime_error(
            "dtl::find requires an explicit communicator when num_ranks()>1; "
            "use dtl::global_find(..., comm) for collective semantics or dtl::local_find(...) for local semantics.");
    }

    using value_type = typename Container::value_type;
    find_result<value_type> result;

    // Search local partition
    auto local_view = container.local_view();
    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (*it == value) {
            result.found = true;
            result.global_index = local_idx;  // Local index (no global offset without communicator)
            result.owner_rank = 0;  // Single-rank mode: always rank 0
            result.value = *it;
            break;
        }
    }

    // Standalone mode: search local partition only.
    // For distributed find across all ranks, use the overload with communicator parameter.
    return result;
}

/// @brief Find first element equal to value (default execution)
template <typename Container, typename T>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("find(container, value) is not collective for multi-rank containers; use find(..., comm)/global_find(..., comm) for global semantics or local_find(...) for local semantics")
auto find(const Container& container, const T& value) {
    return find(seq{}, container, value);
}

// ============================================================================
// Communicator-Aware Distributed Find
// ============================================================================

/// @brief Find first element equal to value with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param value Value to search for
/// @param comm The communicator for allreduce
/// @return Result containing global find information
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result identifies the first occurrence globally (lowest index).
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
find_result<typename Container::value_type>
find([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, const T& value, Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::find");
        !deterministic_guard) {
        throw std::runtime_error(deterministic_guard.error().message());
    }

    using value_type = typename Container::value_type;
    find_result<value_type> local_result;

    // Search local partition
    auto local_view = container.local_view();
    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (*it == value) {
            local_result.found = true;
            local_result.global_index = container.to_global(local_idx);
            local_result.owner_rank = comm.rank();
            local_result.value = *it;
            break;
        }
    }

    // If not found locally, set global_index to max value for min reduction
    index_t my_global_idx = local_result.found ?
        local_result.global_index : std::numeric_limits<index_t>::max();

    // Allreduce to find minimum global index across all ranks
    index_t min_global_idx = comm.template allreduce_min_value<index_t>(my_global_idx);

    find_result<value_type> result;
    if (min_global_idx == std::numeric_limits<index_t>::max()) {
        // Not found on any rank
        result.found = false;
        return result;
    }

    // Found somewhere - determine owner rank
    result.found = true;
    result.global_index = min_global_idx;

    // The rank that found the minimum index broadcasts its result
    if (local_result.found && local_result.global_index == min_global_idx) {
        result.owner_rank = comm.rank();
        result.value = local_result.value;
    }

    // Broadcast the owner rank
    rank_t owner = (local_result.found && local_result.global_index == min_global_idx) ?
        comm.rank() : comm.size();  // Use size() as sentinel for "not me"
    result.owner_rank = comm.template allreduce_min_value<rank_t>(owner);

    return result;
}

/// @brief Find first element equal to value with communicator (default execution)
template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
auto find(const Container& container, const T& value, Comm& comm) {
    return find(seq{}, container, value, comm);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
find_result<typename Container::value_type>
global_find(ExecutionPolicy&& policy, const Container& container, const T& value, Comm& comm) {
    return find(std::forward<ExecutionPolicy>(policy), container, value, comm);
}

template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
auto global_find(const Container& container, const T& value, Comm& comm) {
    return global_find(seq{}, container, value, comm);
}

// ============================================================================
// Distributed find_if
// ============================================================================

/// @brief Find first element satisfying predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Result containing find information
///
/// @par Example:
/// @code
/// auto result = dtl::find_if(dtl::par{}, vec,
///     [](int x) { return x > 100; });
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
find_result<typename Container::value_type>
find_if([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    if (container.num_ranks() > 1) {
        throw std::runtime_error(
            "dtl::find_if requires an explicit communicator when num_ranks()>1; "
            "use dtl::global_find_if(..., comm) for collective semantics or dtl::local_find_if(...) for local semantics.");
    }

    using value_type = typename Container::value_type;
    find_result<value_type> result;

    auto local_view = container.local_view();
    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (pred(*it)) {
            result.found = true;
            result.global_index = local_idx;  // Local index (no global offset without communicator)
            result.owner_rank = 0;  // Single-rank mode: always rank 0
            result.value = *it;
            break;
        }
    }

    // Standalone mode: search local partition only.
    return result;
}

/// @brief Find first element satisfying predicate (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("find_if(container, pred) is not collective for multi-rank containers; use find_if(..., comm)/global_find_if(..., comm) for global semantics or local_find_if(...) for local semantics")
auto find_if(const Container& container, Predicate pred) {
    return find_if(seq{}, container, std::move(pred));
}

/// @brief Find first element satisfying predicate with distributed reduction
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @param comm The communicator for allreduce
/// @return Result containing global find information
template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
find_result<typename Container::value_type>
find_if([[maybe_unused]] ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::find_if");
        !deterministic_guard) {
        throw std::runtime_error(deterministic_guard.error().message());
    }

    using value_type = typename Container::value_type;
    find_result<value_type> local_result;

    auto local_view = container.local_view();
    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (pred(*it)) {
            local_result.found = true;
            local_result.global_index = container.to_global(local_idx);
            local_result.owner_rank = comm.rank();
            local_result.value = *it;
            break;
        }
    }

    // If not found locally, set global_index to max for min reduction
    index_t my_global_idx = local_result.found ?
        local_result.global_index : std::numeric_limits<index_t>::max();

    // Allreduce to find minimum global index
    index_t min_global_idx = comm.template allreduce_min_value<index_t>(my_global_idx);

    find_result<value_type> result;
    if (min_global_idx == std::numeric_limits<index_t>::max()) {
        result.found = false;
        return result;
    }

    result.found = true;
    result.global_index = min_global_idx;

    // Find owner rank
    rank_t owner = (local_result.found && local_result.global_index == min_global_idx) ?
        comm.rank() : comm.size();
    result.owner_rank = comm.template allreduce_min_value<rank_t>(owner);

    if (local_result.found && local_result.global_index == min_global_idx) {
        result.value = local_result.value;
    }

    return result;
}

/// @brief Find first element satisfying predicate with communicator (default execution)
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
auto find_if(const Container& container, Predicate pred, Comm& comm) {
    return find_if(seq{}, container, std::move(pred), comm);
}

template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
find_result<typename Container::value_type>
global_find_if(ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    return find_if(std::forward<ExecutionPolicy>(policy), container, std::move(pred), comm);
}

template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
auto global_find_if(const Container& container, Predicate pred, Comm& comm) {
    return global_find_if(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Distributed find_if_not
// ============================================================================

/// @brief Find first element not satisfying predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Result containing find information
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
find_result<typename Container::value_type>
find_if_not(ExecutionPolicy&& policy, const Container& container, Predicate pred) {
    return find_if(std::forward<ExecutionPolicy>(policy), container,
                   [&pred](const auto& x) { return !pred(x); });
}

/// @brief Find first element not satisfying predicate (default execution)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("find_if_not(container, pred) is not collective for multi-rank containers; use global_find_if_not(..., comm) for global semantics or local_find_if_not(...) for local semantics")
auto find_if_not(const Container& container, Predicate pred) {
    return find_if_not(seq{}, container, std::move(pred));
}

template <typename ExecutionPolicy, typename Container, typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
find_result<typename Container::value_type>
global_find_if_not(ExecutionPolicy&& policy, const Container& container, Predicate pred, Comm& comm) {
    return global_find_if(std::forward<ExecutionPolicy>(policy), container,
                          [&pred](const auto& x) { return !pred(x); }, comm);
}

template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
auto global_find_if_not(const Container& container, Predicate pred, Comm& comm) {
    return global_find_if_not(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// Local-only find (no communication)
// ============================================================================

/// @brief Find in local partition only (no communication)
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container The distributed container
/// @param value Value to search for
/// @return Local iterator to found element, or local end
///
/// @note This is NOT a collective operation. Each rank searches independently.
template <typename Container, typename T>
    requires DistributedContainer<Container>
auto local_find(Container& container, const T& value) {
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (*it == value) {
            return it;
        }
    }
    return local_view.end();
}

/// @brief Find in local partition with predicate (no communication)
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto local_find_if(Container& container, Predicate pred) {
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            return it;
        }
    }
    return local_view.end();
}

}  // namespace dtl
