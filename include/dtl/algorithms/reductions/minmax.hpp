// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file minmax.hpp
/// @brief Distributed min/max algorithms
/// @details Find minimum and maximum elements in distributed containers.
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
#include <dtl/policies/execution/async.hpp>

#include <limits>
#include <optional>
#include <functional>

// Forward declare Communicator concept for disambiguation
#include <dtl/backend/concepts/communicator.hpp>

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

namespace dtl {

// ============================================================================
// Min/Max Result Type
// ============================================================================

/// @brief Result of a distributed min/max operation
/// @tparam T Value type
template <typename T>
struct minmax_result {
    /// @brief The minimum/maximum value found
    T value;

    /// @brief Global index of the element
    index_t global_index = 0;

    /// @brief Rank that owns the element
    rank_t owner_rank = no_rank;

    /// @brief Whether the result is valid (container was non-empty)
    bool valid = false;

    /// @brief Check validity
    explicit operator bool() const noexcept { return valid; }
};

/// @brief Combined min and max result
template <typename T>
struct minmax_pair_result {
    minmax_result<T> min;
    minmax_result<T> max;

    /// @brief Check validity
    explicit operator bool() const noexcept { return min.valid && max.valid; }
};

// ============================================================================
// Distributed min_element
// ============================================================================

/// @brief Find minimum element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @return Result containing minimum value and location
///
/// @par Complexity:
/// O(n/p) local comparisons, plus O(log p) allreduce communication.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto result = dtl::min_element(dtl::par{}, vec);
/// if (result) {
///     std::cout << "Min value: " << result.value
///               << " at index " << result.global_index << "\n";
/// }
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
minmax_result<typename Container::value_type>
min_element([[maybe_unused]] ExecutionPolicy&& policy,
            const Container& container,
            Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::min_element");

    using value_type = typename Container::value_type;
    minmax_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        return result;  // Empty local partition
    }

    result.valid = true;
    result.value = *local_view.begin();
    result.global_index = 0;
    result.owner_rank = 0;  // Not yet implemented: should use comm.rank() for the local rank

    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (comp(*it, result.value)) {
            result.value = *it;
            result.global_index = local_idx;  // Not yet implemented: local-to-global index conversion requires partition offset
        }
    }

    // Limitation: returns local minimum only; global minimum requires allreduce with custom min operation
    return result;
}

/// @brief Find minimum element with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto min_element(const Container& container, Compare comp = Compare{}) {
    return min_element(seq{}, container, std::move(comp));
}

/// @brief Find minimum element with MPI communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @param comm The MPI communicator adapter
/// @return Result containing global minimum value and location
///
/// @par Complexity:
/// O(n/p) local comparisons, plus O(log p) allreduce communication.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<int> vec(1000, comm);
/// auto result = dtl::min_element(dtl::par{}, vec, std::less<>{}, comm);
/// if (result) {
///     std::cout << "Global min value: " << result.value
///               << " at global index " << result.global_index
///               << " owned by rank " << result.owner_rank << "\n";
/// }
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
minmax_result<typename Container::value_type>
min_element([[maybe_unused]] ExecutionPolicy&& policy,
            const Container& container,
            Compare comp,
            Comm& comm) {
    using value_type = typename Container::value_type;
    minmax_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        // Empty local partition - use sentinel values
        result.valid = false;
        result.value = std::numeric_limits<value_type>::max();
        result.global_index = 0;
        result.owner_rank = no_rank;
    } else {
        result.valid = true;
        result.value = *local_view.begin();
        result.global_index = container.global_offset();
        result.owner_rank = comm.rank();

        index_t local_idx = 0;
        for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
            if (comp(*it, result.value)) {
                result.value = *it;
                result.global_index = container.global_offset() + local_idx;
            }
        }
    }

    // Phase 2: Perform distributed min reduction
    // We need to find global min value and track which rank owns it
    // Strategy: Use two allreduces - one for min value, one to find owner rank

    // First allreduce: find global minimum value
    value_type global_min = comm.template allreduce_min_value<value_type>(result.value);

    // Second step: determine which rank owns the global min
    // Each rank reports 1 if it has the global min, 0 otherwise
    // Then we find the first rank (minimum rank number) that has it
    rank_t owner_candidate = (result.valid && result.value == global_min) ? comm.rank() : comm.size();
    rank_t global_owner = comm.template allreduce_min_value<rank_t>(owner_candidate);

    // Third step: broadcast the global index from the owner rank
    index_t global_idx = (comm.rank() == global_owner) ? result.global_index : 0;
    comm.broadcast(&global_idx, sizeof(index_t), global_owner);

    // Construct final result
    minmax_result<value_type> final_result;
    final_result.valid = (global_owner != comm.size());  // Valid if someone had data
    final_result.value = global_min;
    final_result.global_index = global_idx;
    final_result.owner_rank = final_result.valid ? global_owner : no_rank;

    return final_result;
}

// ============================================================================
// Distributed max_element
// ============================================================================

/// @brief Find maximum element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @return Result containing maximum value and location
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
minmax_result<typename Container::value_type>
max_element(ExecutionPolicy&& policy,
            const Container& container,
            Compare comp = Compare{}) {
    // max_element is min_element with inverted comparison
    return min_element(std::forward<ExecutionPolicy>(policy), container,
                       [&comp](const auto& a, const auto& b) { return comp(b, a); });
}

/// @brief Find maximum element with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto max_element(const Container& container, Compare comp = Compare{}) {
    return max_element(seq{}, container, std::move(comp));
}

/// @brief Find maximum element with MPI communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @param comm The MPI communicator adapter
/// @return Result containing global maximum value and location
///
/// @note Unlike the non-communicator version, this implementation does NOT
///       delegate to min_element with inverted comparison because the
///       distributed reduction requires allreduce_max_value (not min).
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
minmax_result<typename Container::value_type>
max_element([[maybe_unused]] ExecutionPolicy&& policy,
            const Container& container,
            Compare comp,
            Comm& comm) {
    using value_type = typename Container::value_type;
    minmax_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        // Empty local partition - use sentinel values for max search
        result.valid = false;
        result.value = std::numeric_limits<value_type>::lowest();
        result.global_index = 0;
        result.owner_rank = no_rank;
    } else {
        result.valid = true;
        result.value = *local_view.begin();
        result.global_index = container.global_offset();
        result.owner_rank = comm.rank();

        index_t local_idx = 0;
        for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
            // comp(result.value, *it) means result.value < *it, so *it is larger
            if (comp(result.value, *it)) {
                result.value = *it;
                result.global_index = container.global_offset() + local_idx;
            }
        }
    }

    // Phase 2: Perform distributed max reduction
    // We need to find global max value and track which rank owns it

    // First allreduce: find global maximum value
    value_type global_max = comm.template allreduce_max_value<value_type>(result.value);

    // Second step: determine which rank owns the global max
    // Each rank reports its rank if it has the global max, size() otherwise
    // Then we find the first rank (minimum rank number) that has it
    rank_t owner_candidate = (result.valid && result.value == global_max) ? comm.rank() : comm.size();
    rank_t global_owner = comm.template allreduce_min_value<rank_t>(owner_candidate);

    // Third step: broadcast the global index from the owner rank
    index_t global_idx = (comm.rank() == global_owner) ? result.global_index : 0;
    comm.broadcast(&global_idx, sizeof(index_t), global_owner);

    // Construct final result
    minmax_result<value_type> final_result;
    final_result.valid = (global_owner != comm.size());  // Valid if someone had data
    final_result.value = global_max;
    final_result.global_index = global_idx;
    final_result.owner_rank = final_result.valid ? global_owner : no_rank;

    return final_result;
}

// ============================================================================
// Distributed minmax_element
// ============================================================================

/// @brief Find both minimum and maximum elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @return Pair containing both min and max results
///
/// @par Efficiency:
/// Single pass over data, finding both min and max together.
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
minmax_pair_result<typename Container::value_type>
minmax_element([[maybe_unused]] ExecutionPolicy&& policy,
               const Container& container,
               Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::minmax_element");

    using value_type = typename Container::value_type;
    minmax_pair_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        return result;
    }

    result.min.valid = result.max.valid = true;
    result.min.value = result.max.value = *local_view.begin();
    result.min.global_index = result.max.global_index = 0;
    result.min.owner_rank = result.max.owner_rank = 0;

    index_t local_idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
        if (comp(*it, result.min.value)) {
            result.min.value = *it;
            result.min.global_index = local_idx;
        }
        if (comp(result.max.value, *it)) {
            result.max.value = *it;
            result.max.global_index = local_idx;
        }
    }

    // Limitation: returns local minmax only; global minmax requires two allreduces (one for min, one for max)
    return result;
}

/// @brief Find minmax with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto minmax_element(const Container& container, Compare comp = Compare{}) {
    return minmax_element(seq{}, container, std::move(comp));
}

/// @brief Find both minimum and maximum elements with MPI communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return Pair containing both global min and max results
///
/// @par Efficiency:
/// Single pass over data, finding both min and max together.
/// Uses two allreduce operations for min and max separately.
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
minmax_pair_result<typename Container::value_type>
minmax_element([[maybe_unused]] ExecutionPolicy&& policy,
               const Container& container,
               Compare comp,
               Comm& comm) {
    using value_type = typename Container::value_type;
    minmax_pair_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        // Empty local partition
        result.min.valid = result.max.valid = false;
        result.min.value = std::numeric_limits<value_type>::max();
        result.max.value = std::numeric_limits<value_type>::lowest();
        result.min.global_index = result.max.global_index = 0;
        result.min.owner_rank = result.max.owner_rank = no_rank;
    } else {
        result.min.valid = result.max.valid = true;
        result.min.value = result.max.value = *local_view.begin();
        result.min.global_index = result.max.global_index = container.global_offset();
        result.min.owner_rank = result.max.owner_rank = comm.rank();

        index_t local_idx = 0;
        for (auto it = local_view.begin(); it != local_view.end(); ++it, ++local_idx) {
            if (comp(*it, result.min.value)) {
                result.min.value = *it;
                result.min.global_index = container.global_offset() + local_idx;
            }
            if (comp(result.max.value, *it)) {
                result.max.value = *it;
                result.max.global_index = container.global_offset() + local_idx;
            }
        }
    }

    // Perform distributed min and max reductions
    // Min reduction
    value_type global_min = comm.template allreduce_min_value<value_type>(result.min.value);
    rank_t min_owner_candidate = (result.min.valid && result.min.value == global_min)
                                   ? comm.rank() : comm.size();
    rank_t global_min_owner = comm.template allreduce_min_value<rank_t>(min_owner_candidate);
    index_t global_min_idx = (comm.rank() == global_min_owner) ? result.min.global_index : 0;
    comm.broadcast(&global_min_idx, sizeof(index_t), global_min_owner);

    // Max reduction
    value_type global_max = comm.template allreduce_max_value<value_type>(result.max.value);
    rank_t max_owner_candidate = (result.max.valid && result.max.value == global_max)
                                   ? comm.rank() : comm.size();
    rank_t global_max_owner = comm.template allreduce_min_value<rank_t>(max_owner_candidate);
    index_t global_max_idx = (comm.rank() == global_max_owner) ? result.max.global_index : 0;
    comm.broadcast(&global_max_idx, sizeof(index_t), global_max_owner);

    // Construct final result
    minmax_pair_result<value_type> final_result;
    final_result.min.valid = (global_min_owner != comm.size());
    final_result.min.value = global_min;
    final_result.min.global_index = global_min_idx;
    final_result.min.owner_rank = final_result.min.valid ? global_min_owner : no_rank;

    final_result.max.valid = (global_max_owner != comm.size());
    final_result.max.value = global_max;
    final_result.max.global_index = global_max_idx;
    final_result.max.owner_rank = final_result.max.valid ? global_max_owner : no_rank;

    return final_result;
}

// ============================================================================
// Value-only min/max (no location)
// ============================================================================

/// @brief Get minimum value only
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Minimum value (or empty optional if container is empty)
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
std::optional<typename Container::value_type>
min(ExecutionPolicy&& policy, const Container& container) {
    auto result = min_element(std::forward<ExecutionPolicy>(policy), container);
    if (result) {
        return result.value;
    }
    return std::nullopt;
}

/// @brief Get maximum value only
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
std::optional<typename Container::value_type>
max(ExecutionPolicy&& policy, const Container& container) {
    auto result = max_element(std::forward<ExecutionPolicy>(policy), container);
    if (result) {
        return result.value;
    }
    return std::nullopt;
}

/// @brief Get both min and max values
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto minmax(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    struct minmax_values {
        std::optional<value_type> min;
        std::optional<value_type> max;
    };

    auto result = minmax_element(std::forward<ExecutionPolicy>(policy), container);
    minmax_values values;
    if (result) {
        values.min = result.min.value;
        values.max = result.max.value;
    }
    return values;
}

// ============================================================================
// Local-only min/max (no communication)
// ============================================================================

/// @brief Find minimum in local partition only
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param container The distributed container
/// @param comp Comparison function
/// @return Local minimum result
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto local_min_element(const Container& container, Compare comp = Compare{}) {
    using value_type = typename Container::value_type;
    minmax_result<value_type> result;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        return result;
    }

    result.valid = true;
    result.value = *local_view.begin();
    result.global_index = 0;

    index_t idx = 0;
    for (auto it = local_view.begin(); it != local_view.end(); ++it, ++idx) {
        if (comp(*it, result.value)) {
            result.value = *it;
            result.global_index = idx;
        }
    }

    return result;
}

/// @brief Find maximum in local partition only
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto local_max_element(const Container& container, Compare comp = Compare{}) {
    return local_min_element(container,
        [&comp](const auto& a, const auto& b) { return comp(b, a); });
}

// ============================================================================
// Async min/max
// ============================================================================

/// @brief Asynchronously find minimum
template <typename Container>
    requires DistributedContainer<Container>
auto async_min_element(const Container& container)
    -> futures::distributed_future<minmax_result<typename Container::value_type>> {
    using value_type = typename Container::value_type;
    auto promise = std::make_shared<futures::distributed_promise<minmax_result<value_type>>>();
    auto future = promise->get_future();

    try {
        auto result = min_element(seq{}, container, std::less<>{});
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously find maximum
template <typename Container>
    requires DistributedContainer<Container>
auto async_max_element(const Container& container)
    -> futures::distributed_future<minmax_result<typename Container::value_type>> {
    using value_type = typename Container::value_type;
    auto promise = std::make_shared<futures::distributed_promise<minmax_result<value_type>>>();
    auto future = promise->get_future();

    try {
        auto result = max_element(seq{}, container, std::less<>{});
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously find both minimum and maximum
template <typename Container>
    requires DistributedContainer<Container>
auto async_minmax_element(const Container& container)
    -> futures::distributed_future<minmax_pair_result<typename Container::value_type>> {
    using value_type = typename Container::value_type;
    auto promise = std::make_shared<futures::distributed_promise<minmax_pair_result<value_type>>>();
    auto future = promise->get_future();

    try {
        auto result = minmax_element(seq{}, container);
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
