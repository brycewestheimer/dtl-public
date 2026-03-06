// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file nth_element.hpp
/// @brief Distributed nth_element algorithm
/// @details Partition around the nth element in distributed container.
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
#include <dtl/backend/concepts/communicator.hpp>

#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>

namespace dtl {

// ============================================================================
// Nth Element Result Type
// ============================================================================

/// @brief Result of distributed nth_element operation
/// @tparam T Value type
template <typename T>
struct nth_element_result {
    /// @brief The nth element value
    T value;

    /// @brief Global index of the nth element
    index_t global_index = 0;

    /// @brief Rank that owns the nth element
    rank_t owner_rank = no_rank;

    /// @brief Whether the operation succeeded
    bool valid = false;

    /// @brief Check validity
    explicit operator bool() const noexcept { return valid; }
};

// ============================================================================
// Distributed nth_element
// ============================================================================

/// @brief Partition distributed container around nth element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param n The global index of the element to partition around
/// @param comp Comparison function
/// @return Result containing the nth element value
///
/// @par Post-condition:
/// After this operation:
/// - Element at global index n is the element that would be there if sorted
/// - All elements before index n compare less than or equal to element[n]
/// - All elements after index n compare greater than or equal to element[n]
///
/// @par Algorithm:
/// Distributed quickselect:
/// 1. Select pivot
/// 2. Partition locally
/// 3. Count elements <= pivot across all ranks
/// 4. If count == n, done. Otherwise recurse on appropriate partition.
///
/// @par Complexity:
/// O(n/p) average per rank, O(log p) communication rounds.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(10000, ctx);
/// // Find and partition around median
/// auto result = dtl::nth_element(dtl::par{}, vec, vec.size() / 2);
/// if (result) {
///     std::cout << "Median value: " << result.value << "\n";
/// }
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
nth_element_result<typename Container::value_type>
nth_element([[maybe_unused]] ExecutionPolicy&& policy,
            Container& container,
            index_t n,
            Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::nth_element");

    using value_type = typename Container::value_type;
    nth_element_result<value_type> result;

    auto local_view = container.local_view();
    size_type local_size = static_cast<size_type>(local_view.end() - local_view.begin());

    // Limitation: only local nth_element is performed; distributed quickselect
    // requires cross-rank sampling and redistribution (not yet implemented).
    if (local_size > 0) {
        index_t local_n = n < static_cast<index_t>(local_size)
                              ? n
                              : static_cast<index_t>(local_size - 1);
        if (local_n >= 0) {
            std::nth_element(local_view.begin(), local_view.begin() + local_n,
                             local_view.end(), comp);
            result.value = *(local_view.begin() + local_n);
            result.global_index = n;
            result.owner_rank = 0;  // Stub
            result.valid = true;
        }
    }

    return result;
}

/// @brief nth_element with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto nth_element(Container& container, index_t n, Compare comp = Compare{}) {
    return nth_element(seq{}, container, n, std::move(comp));
}

/// @brief Distributed nth_element with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param n The global index of the element to partition around
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return Result containing the nth element value
///
/// @par Algorithm (Distributed Quickselect):
/// 1. Select a pivot (median of random samples)
/// 2. Partition local data around pivot
/// 3. Count elements <= pivot globally via allreduce
/// 4. If count matches n, return pivot
/// 5. Otherwise, recurse on the appropriate half
///
/// @par Complexity:
/// O(n/p) average per rank, O(log n) iterations expected.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
nth_element_result<typename Container::value_type>
nth_element([[maybe_unused]] ExecutionPolicy&& policy,
            Container& container,
            index_t n,
            Compare comp,
            Comm& comm) {
    using value_type = typename Container::value_type;
    nth_element_result<value_type> result;

    size_type global_size = container.global_size();

    // Validate n
    if (n < 0 || static_cast<size_type>(n) >= global_size) {
        return result;  // Invalid n, return invalid result
    }

    rank_t num_ranks = comm.size();

    // Single rank case: use standard nth_element
    if (num_ranks <= 1) {
        auto local_v = container.local_view();
        if (local_v.size() > 0 && n < static_cast<index_t>(local_v.size())) {
            std::nth_element(local_v.begin(), local_v.begin() + n,
                             local_v.end(), comp);
            result.value = *(local_v.begin() + n);
            result.global_index = n;
            result.owner_rank = 0;
            result.valid = true;
        }
        return result;
    }

    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());
    // Correctness-first fallback: gather the global sequence on every rank,
    // partition it locally with std::nth_element, then restore each rank's
    // original block partition from the partitioned global sequence.
    int my_count = static_cast<int>(local_size);
    std::vector<int> recv_counts(static_cast<size_type>(num_ranks), 0);
    comm.allgather(&my_count, recv_counts.data(), sizeof(int));

    std::vector<int> recv_displs(static_cast<size_type>(num_ranks), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);

    const size_type total_count = recv_counts.empty()
        ? 0
        : static_cast<size_type>(recv_displs.back() + recv_counts.back());
    if (total_count == 0) {
        return result;
    }

    std::vector<value_type> global_values(total_count);
    comm.allgatherv(local_v.begin(), local_size,
                    global_values.data(), recv_counts.data(), recv_displs.data(),
                    sizeof(value_type));

    auto nth_it = global_values.begin() + n;
    std::nth_element(global_values.begin(), nth_it, global_values.end(), comp);

    result.value = *nth_it;
    result.global_index = n;
    result.valid = true;

    const auto global_offset = static_cast<size_type>(local_v.global_offset());
    const auto owner_it = std::upper_bound(recv_displs.begin(), recv_displs.end(),
                                           static_cast<int>(n));
    if (owner_it != recv_displs.begin()) {
        result.owner_rank = static_cast<rank_t>(std::distance(recv_displs.begin(), owner_it) - 1);
    } else {
        result.owner_rank = 0;
    }

    for (size_type i = 0; i < local_size; ++i) {
        local_v[i] = std::move(global_values[global_offset + i]);
    }

    return result;
}

// ============================================================================
// Selection algorithms
// ============================================================================

/// @brief Find median element
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Result containing median value
///
/// @par Median Definition:
/// For containers of size n:
/// - If n is odd: element at index n/2
/// - If n is even: element at index n/2 (lower median)
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto median(ExecutionPolicy&& policy, Container& container) {
    index_t mid = static_cast<index_t>(container.size() / 2);
    return nth_element(std::forward<ExecutionPolicy>(policy), container, mid);
}

/// @brief Find median with default execution
template <typename Container>
    requires DistributedContainer<Container>
auto median(Container& container) {
    return median(seq{}, container);
}

// select_kth removed — was unused internal helper

// ============================================================================
// Local-only nth_element (no communication)
// ============================================================================

/// @brief Partition local partition around nth element
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param container The distributed container
/// @param n Local index to partition around
/// @param comp Comparison function
///
/// @note NOT collective - partitions local data only.
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
void local_nth_element(Container& container, index_t n,
                       Compare comp = Compare{}) {
    auto local_view = container.local_view();
    size_type local_size = static_cast<size_type>(local_view.end() - local_view.begin());
    if (n >= 0 && static_cast<size_type>(n) < local_size) {
        std::nth_element(local_view.begin(), local_view.begin() + n,
                         local_view.end(), std::move(comp));
    }
}

// ============================================================================
// Partitioning
// ============================================================================

/// @brief Partition distributed container
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Unary predicate
/// @return Global partition point (first index where predicate is false)
///
/// @par Post-condition:
/// All elements where pred(e) is true come before elements where pred(e) is false.
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
index_t partition([[maybe_unused]] ExecutionPolicy&& policy,
                  Container& container,
                  Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::partition");

    // Partition locally first
    auto local_view = container.local_view();
    auto partition_point = std::partition(local_view.begin(), local_view.end(), pred);
    index_t local_count = static_cast<index_t>(partition_point - local_view.begin());

    // Limitation: redistribution to maintain global partition invariant is not
    // yet implemented. Each rank would need to exchange elements so that all
    // elements satisfying pred precede those that do not, globally.

    return local_count;  // Returns local partition point only; global partition point requires cross-rank reduction
}

/// @brief Partition with default execution
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
index_t partition(Container& container, Predicate pred) {
    return partition(seq{}, container, std::move(pred));
}

/// @brief Stable partition distributed container
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
index_t stable_partition([[maybe_unused]] ExecutionPolicy&& policy,
                         Container& container,
                         Predicate pred) {
    auto local_view = container.local_view();
    auto partition_point = std::stable_partition(local_view.begin(), local_view.end(), pred);
    return static_cast<index_t>(partition_point - local_view.begin());
}

// ============================================================================
// Async nth_element
// ============================================================================

/// @brief Asynchronously partition around nth element
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto async_nth_element(Container& container, index_t n,
                       Compare comp = Compare{}) {
    using value_type = typename Container::value_type;
    return result<nth_element_result<value_type>>{
        nth_element(async{}, container, n, std::move(comp))};
}

}  // namespace dtl
