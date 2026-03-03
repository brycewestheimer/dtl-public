// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file partition_algorithm.hpp
/// @brief Distributed partition algorithm
/// @details Partition elements of a distributed container so that elements
///          satisfying a predicate precede those that do not.
///          Named partition_algorithm.hpp to avoid collision with
///          dtl/policies/partition/ directory.
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

#include <algorithm>
#include <memory>

namespace dtl {

// ============================================================================
// Partition Result Type
// ============================================================================

/// @brief Result of a distributed partition operation
struct partition_result {
    /// @brief Number of elements satisfying the predicate (local count)
    size_type local_true_count = 0;

    /// @brief Number of elements satisfying the predicate (global count)
    /// Only valid when has_global is true.
    size_type global_true_count = 0;

    /// @brief Whether global_true_count is valid (communicator was used)
    bool has_global = false;
};

// ============================================================================
// Local Partition (within rank's partition)
// ============================================================================

/// @brief Partition local elements so that those satisfying predicate come first
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @param policy Execution policy
/// @param container Container to partition
/// @param pred Predicate function
/// @return partition_result with local_true_count
///
/// @par Complexity:
/// O(n/p) local element moves. No communication required.
///
/// @note This partitions ONLY within each rank's local data.
///       The relative order of elements may not be preserved (unstable partition).
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(100, ctx);
/// auto result = dtl::partition_elements(dtl::seq{}, vec,
///     [](int x) { return x % 2 == 0; });
/// // Local elements: [even..., odd...]
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
partition_result partition_elements([[maybe_unused]] ExecutionPolicy&& policy,
                                   Container& container,
                                   Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::partition_elements");

    auto local_v = container.local_view();

    auto pivot = std::partition(local_v.begin(), local_v.end(), pred);
    size_type true_count = static_cast<size_type>(
        std::distance(local_v.begin(), pivot));

    return partition_result{true_count, 0, false};
}

/// @brief Partition with default execution
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("partition_elements(container, pred) is local-only for multi-rank containers; use partition_elements(..., comm) for collective semantics")
partition_result partition_elements(Container& container, Predicate pred) {
    return partition_elements(seq{}, container, std::move(pred));
}

// ============================================================================
// Stable Partition (preserves relative order)
// ============================================================================

/// @brief Stable partition: preserves relative order within each group
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @param policy Execution policy
/// @param container Container to partition
/// @param pred Predicate function
/// @return partition_result with local_true_count
///
/// @par Complexity:
/// O(n/p) local element moves. No communication required.
///
/// @note Relative order of elements within the "true" and "false" groups
///       is preserved within each rank's local partition.
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
partition_result stable_partition_elements([[maybe_unused]] ExecutionPolicy&& policy,
                                          Container& container,
                                          Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::stable_partition_elements");

    auto local_v = container.local_view();

    auto pivot = std::stable_partition(local_v.begin(), local_v.end(), pred);
    size_type true_count = static_cast<size_type>(
        std::distance(local_v.begin(), pivot));

    return partition_result{true_count, 0, false};
}

/// @brief Stable partition with default execution
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("stable_partition_elements(container, pred) is local-only for multi-rank containers; use communicator-aware partition flows for distributed semantics")
partition_result stable_partition_elements(Container& container, Predicate pred) {
    return stable_partition_elements(seq{}, container, std::move(pred));
}

// ============================================================================
// Partition with Communicator (global count)
// ============================================================================

/// @brief Partition with global count via communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container Container to partition
/// @param pred Predicate function
/// @param comm Communicator for allreduce of counts
/// @return partition_result with both local and global counts
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The partition is performed locally, then an allreduce provides
/// the global count of elements satisfying the predicate.
template <typename ExecutionPolicy, typename Container,
          typename Predicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
partition_result partition_elements(ExecutionPolicy&& policy,
                                   Container& container,
                                   Predicate pred,
                                   Comm& comm) {
    // Phase 1: Local partition
    auto local_res = partition_elements(
        std::forward<ExecutionPolicy>(policy), container, std::move(pred));

    // Phase 2: Allreduce to get global count
    size_type global_count = comm.template allreduce_sum_value<size_type>(
        local_res.local_true_count);

    return partition_result{local_res.local_true_count, global_count, true};
}

/// @brief Partition with communicator and default execution
template <typename Container, typename Predicate, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
partition_result partition_elements(Container& container, Predicate pred, Comm& comm) {
    return partition_elements(seq{}, container, std::move(pred), comm);
}

// ============================================================================
// is_partitioned (check if already partitioned)
// ============================================================================

/// @brief Check if local elements are already partitioned by predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @param policy Execution policy
/// @param container Container to check
/// @param pred Predicate function
/// @return true if local elements satisfying pred precede those that don't
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool is_partitioned([[maybe_unused]] ExecutionPolicy&& policy,
                    const Container& container,
                    Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::is_partitioned");

    auto local_v = container.local_view();
    return std::is_partitioned(local_v.begin(), local_v.end(), std::move(pred));
}

/// @brief Check partition with default execution
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("is_partitioned(container, pred) is local-only for multi-rank containers; use communicator-aware checks for global semantics")
bool is_partitioned(const Container& container, Predicate pred) {
    return is_partitioned(seq{}, container, std::move(pred));
}

// ============================================================================
// partition_count (count without modifying)
// ============================================================================

/// @brief Count elements satisfying predicate without partitioning
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @param policy Execution policy
/// @param container Container to count
/// @param pred Predicate function
/// @return Local count of elements satisfying pred
template <typename ExecutionPolicy, typename Container, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type partition_count([[maybe_unused]] ExecutionPolicy&& policy,
                          const Container& container,
                          Predicate pred) {
    detail::require_collective_comm_or_single_rank(container, "dtl::partition_count");

    auto local_v = container.local_view();
    return static_cast<size_type>(
        std::count_if(local_v.begin(), local_v.end(), std::move(pred)));
}

/// @brief Partition count with default execution
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("partition_count(container, pred) is local-only for multi-rank containers; use partition_elements(..., comm) for collective global counts")
size_type partition_count(const Container& container, Predicate pred) {
    return partition_count(seq{}, container, std::move(pred));
}

// ============================================================================
// Async Partition
// ============================================================================

/// @brief Asynchronously partition local elements
/// @tparam Container Distributed container type
/// @tparam Predicate Unary predicate type
/// @param container Container to partition
/// @param pred Predicate function
/// @return Future containing partition_result
template <typename Container, typename Predicate>
    requires DistributedContainer<Container>
auto async_partition_elements(Container& container, Predicate pred)
    -> futures::distributed_future<partition_result> {
    auto promise = std::make_shared<futures::distributed_promise<partition_result>>();
    auto future = promise->get_future();

    try {
        auto res = partition_elements(seq{}, container, std::move(pred));
        promise->set_value(res);
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
