// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file sort.hpp
/// @brief Distributed sort algorithm
/// @details Sort elements across distributed container using sample sort.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/algorithms/sorting/sort_types.hpp>
#include <dtl/algorithms/sorting/sample_sort_detail.hpp>
#include <dtl/algorithms/sorting/stable_sort_global.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/backend/concepts/communicator.hpp>

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <algorithm>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

namespace dtl {

// ============================================================================
// Distributed sort (sample sort algorithm)
// ============================================================================

/// @brief Sort distributed container globally (standalone, single-rank version)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @return Result indicating success or failure
///
/// @par Algorithm:
/// For single-rank or standalone usage, performs local sort only.
/// For multi-rank usage, use the overload with communicator parameter.
///
/// @par Complexity:
/// O(n log n) local sort.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(10000, ctx);
/// dtl::sort(dtl::par{}, vec);  // Local sort only
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> sort(ExecutionPolicy&& policy,
                  Container& container,
                  Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::sort");

    // Single-rank: local sort is sufficient
    auto local_v = container.local_view();
    dispatch_sort(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.end(), comp);
    return {};
}

// ============================================================================
// Full Sample Sort with MPI Communicator
// ============================================================================

/// @brief Sort distributed container globally using sample sort with MPI
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @param comm The MPI communicator adapter
/// @return distributed_sort_result with statistics
///
/// @par Algorithm (8 Phases):
/// 1. Local sort on each rank
/// 2. Sample local data (evenly-spaced elements)
/// 3. Allgather samples from all ranks
/// 4. Select p-1 global pivots from gathered samples
/// 5. Partition local data into buckets by pivots
/// 6. Compute alltoallv parameters and exchange counts
/// 7. Flatten buckets and exchange data via alltoallv
/// 8. Merge received sorted chunks
///
/// @par Complexity:
/// O((n/p) log(n/p)) local sort + O(p * samples) allgather + O(n/p) alltoallv.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Post-condition:
/// Container is globally sorted. Element at global index i is less than
/// or equal to element at global index i+1 for all valid i.
/// Note: Local sizes may change after sort due to data redistribution.
///
/// @par Example:
/// @code
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<int> vec(10000, comm);
/// dtl::sort(dtl::par{}, vec, std::less<>{}, comm);  // Globally sorted
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
distributed_sort_result sort(ExecutionPolicy&& policy,
                              Container& container,
                              Compare comp,
                              Comm& comm) {
    using value_type = typename Container::value_type;
    distributed_sort_result result{true, 0, 0};

    rank_t num_ranks = comm.size();
    [[maybe_unused]] rank_t my_rank = comm.rank();

    // Handle single rank case: just local sort
    if (num_ranks <= 1) {
        auto local_v = container.local_view();
        dispatch_sort(std::forward<ExecutionPolicy>(policy),
                      local_v.begin(), local_v.end(), comp);
        return result;
    }

    // ========================================================================
    // Phase 1: Local sort
    // ========================================================================
    auto local_v = container.local_view();
    dispatch_sort(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.end(), comp);

    size_type local_size = container.local_size();

    // Handle empty local partition
    if (container.global_size() == 0) {
        return result;
    }

    // ========================================================================
    // Phase 2: Sample local data
    // ========================================================================
    // Take oversampling_factor * num_ranks samples for good pivot quality
    size_type samples_per_rank = std::max(
        static_cast<size_type>(1),
        std::min(local_size, static_cast<size_type>(3 * num_ranks)));

    auto local_samples = detail::sample_local(
        local_v.begin(), local_v.end(), samples_per_rank, comp);

    // ========================================================================
    // Phase 3: Allgather samples to all ranks
    // ========================================================================
    // Gather sample counts first (may vary per rank)
    int my_sample_count = static_cast<int>(local_samples.size());
    std::vector<int> sample_counts(static_cast<size_type>(num_ranks));
    comm.allgather(&my_sample_count, sample_counts.data(), sizeof(int));

    // Compute displacements
    std::vector<int> sample_displs(static_cast<size_type>(num_ranks));
    std::exclusive_scan(sample_counts.begin(), sample_counts.end(),
                        sample_displs.begin(), 0);

    size_type total_samples = static_cast<size_type>(
        sample_displs.back() + sample_counts.back());

    // Allgather the actual samples
    std::vector<value_type> all_samples(total_samples);
    comm.allgatherv(local_samples.data(), static_cast<size_type>(my_sample_count),
                    all_samples.data(), sample_counts.data(), sample_displs.data(),
                    sizeof(value_type));

    // ========================================================================
    // Phase 4: Select p-1 pivots from gathered samples
    // ========================================================================
    auto pivots = detail::select_pivots(all_samples, num_ranks, comp);

    // ========================================================================
    // Phase 5: Partition local data into p buckets by pivots
    // ========================================================================
    auto buckets = detail::partition_by_pivots(
        local_v.begin(), local_v.end(), pivots, comp);

    // ========================================================================
    // Phase 6: Compute alltoallv parameters and exchange counts
    // ========================================================================
    auto params = detail::compute_alltoallv_params(buckets);

    // Exchange send_counts to get recv_counts via alltoall
    comm.alltoall(params.send_counts.data(), params.recv_counts.data(), sizeof(int));

    // Compute receive displacements
    std::exclusive_scan(params.recv_counts.begin(), params.recv_counts.end(),
                        params.recv_displs.begin(), 0);
    params.total_recv = static_cast<size_type>(
        params.recv_displs.back() + params.recv_counts.back());

    // ========================================================================
    // Phase 7: Flatten buckets and exchange data via alltoallv
    // ========================================================================
    auto send_buffer = detail::flatten_buckets(buckets);
    std::vector<value_type> recv_buffer(params.total_recv);

    result.elements_sent = send_buffer.size();
    result.elements_received = params.total_recv;

    comm.alltoallv(send_buffer.data(), params.send_counts.data(), params.send_displs.data(),
                   recv_buffer.data(), params.recv_counts.data(), params.recv_displs.data(),
                   sizeof(value_type));

    // ========================================================================
    // Phase 8: Merge received sorted chunks
    // ========================================================================
    // Each received chunk from each rank is already sorted.
    // We need to merge them into final sorted order.

    // Approach 1: k-way merge (optimal but complex)
    // Approach 2: Simple re-sort (simpler, still efficient for small p)
    // Using approach 2 for now, which is O((n/p) log(n/p))

    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::sort(std::execution::par, recv_buffer.begin(), recv_buffer.end(), comp);
    } else {
        std::sort(recv_buffer.begin(), recv_buffer.end(), comp);
    }

    // ========================================================================
    // Update container with sorted data (structural contract)
    // ========================================================================
    if constexpr (requires(Container& c, typename Container::storage_type data) {
                      c.replace_local_partition(std::move(data));
                  }) {
        typename Container::storage_type new_local_data(
            recv_buffer.begin(), recv_buffer.end());
        auto apply_result = container.replace_local_partition(std::move(new_local_data));
        if (!apply_result) {
            result.success = false;
            return result;
        }
    } else {
        result.success = false;
        return result;
    }

    return result;
}

/// @brief Sort with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("sort(container, comp) is local-only for multi-rank containers; use sort(..., comp, comm) for collective global sort or local_sort(...) for rank-local semantics")
result<void> sort(Container& container, Compare comp = Compare{}) {
    return sort(seq{}, container, std::move(comp));
}

/// @brief Distributed sort with configuration options (standalone version)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param config Sort configuration
/// @return distributed_sort_result with statistics
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
distributed_sort_result distributed_sort(ExecutionPolicy&& policy,
                                          Container& container,
                                          Compare comp = Compare{},
                                          distributed_sort_config config = {}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::distributed_sort");

    // Phase 1: Sort local partition
    auto local_v = container.local_view();

    if (config.use_parallel_local_sort) {
        dispatch_sort(par{}, local_v.begin(), local_v.end(), comp);
    } else {
        dispatch_sort(seq{}, local_v.begin(), local_v.end(), comp);
    }

    // For single-rank, local sort is sufficient
    return distributed_sort_result{true, 0, 0};
}

/// @brief Distributed sort with configuration options and communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param config Sort configuration
/// @param comm The MPI communicator adapter
/// @return distributed_sort_result with statistics
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
distributed_sort_result distributed_sort(ExecutionPolicy&& policy,
                                          Container& container,
                                          Compare comp,
                                          distributed_sort_config config,
                                          Comm& comm) {
    // Use the sample sort implementation
    if (config.use_parallel_local_sort) {
        return sort(par{}, container, comp, comm);
    } else {
        return sort(seq{}, container, comp, comm);
    }
}

// ============================================================================
// Stable sort
// ============================================================================

/// @brief Stable sort distributed container (standalone version)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @return Result indicating success or failure
///
/// @par Stability:
/// Equal elements maintain their relative order from before sorting.
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> stable_sort(ExecutionPolicy&& policy,
                         Container& container,
                         Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::stable_sort");

    auto local_v = container.local_view();

    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::stable_sort(std::execution::par, local_v.begin(), local_v.end(), comp);
    } else {
        std::stable_sort(local_v.begin(), local_v.end(), comp);
    }

    return {};
}

/// @brief Stable sort with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("stable_sort(container, comp) is local-only for multi-rank containers; use stable_sort(..., comp, comm) for collective global stable sort or local_stable_sort(...) for rank-local semantics")
result<void> stable_sort(Container& container, Compare comp = Compare{}) {
    return stable_sort(seq{}, container, std::move(comp));
}

/// @brief Stable sort distributed container with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return distributed_sort_result with statistics
///
/// @par Algorithm:
/// Delegates to stable_sort_global() which uses origin-tagged elements
/// to guarantee global stability. Each element is augmented with
/// (original_rank, original_index) pairs. Equal keys are tie-broken
/// by origin, preserving global ordering.
///
/// @par Stability:
/// Equal elements maintain their original relative ordering across all
/// ranks. This is guaranteed by the origin-tracking approach in
/// stable_sort_global(). Note: ~2x memory overhead due to origin tags.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @see stable_sort_global() in stable_sort_global.hpp for the full
///      implementation with origin tracking.
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
distributed_sort_result stable_sort(ExecutionPolicy&& policy,
                                     Container& container,
                                     Compare comp,
                                     Comm& comm) {
    return stable_sort_global(std::forward<ExecutionPolicy>(policy),
                              container, comp, comm);
}

// ============================================================================
// Local-only sort (no communication)
// ============================================================================

/// @brief Sort local partition only (no communication)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
///
/// @note NOT collective - sorts local data only. Global ordering not guaranteed.
template <typename ExecutionPolicy, typename Container, typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
void local_sort(ExecutionPolicy&& policy, Container& container, Compare comp = Compare{}) {
    auto local_v = container.local_view();
    dispatch_sort(std::forward<ExecutionPolicy>(policy),
                  local_v.begin(), local_v.end(), comp);
}

/// @brief Sort local partition with default sequential execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
void local_sort(Container& container, Compare comp = Compare{}) {
    local_sort(seq{}, container, std::move(comp));
}

/// @brief Stable sort local partition only
template <typename ExecutionPolicy, typename Container, typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
void local_stable_sort(ExecutionPolicy&& policy, Container& container, Compare comp = Compare{}) {
    auto local_v = container.local_view();
    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::stable_sort(std::execution::par, local_v.begin(), local_v.end(), comp);
    } else {
        std::stable_sort(local_v.begin(), local_v.end(), comp);
    }
}

/// @brief Stable sort local partition with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
void local_stable_sort(Container& container, Compare comp = Compare{}) {
    local_stable_sort(seq{}, container, std::move(comp));
}

// ============================================================================
// Is sorted check
// ============================================================================

/// @brief Check if container is globally sorted (standalone version)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @return true if locally sorted (cannot verify boundaries without communicator)
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool is_sorted([[maybe_unused]] ExecutionPolicy&& policy,
               const Container& container,
               Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::is_sorted");

    // Check if local partition is sorted
    auto local_v = container.local_view();
    return std::is_sorted(local_v.begin(), local_v.end(), comp);
}

/// @brief Check if sorted with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("is_sorted(container, comp) is local-only for multi-rank containers; use is_sorted(..., comp, comm) for global sortedness or local_is_sorted(...) for rank-local semantics")
bool is_sorted(const Container& container, Compare comp = Compare{}) {
    return is_sorted(seq{}, container, std::move(comp));
}

/// @brief Check if container is globally sorted with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return true if globally sorted (including boundary checks)
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Algorithm:
/// 1. Check if local partition is sorted
/// 2. Exchange boundary elements with neighboring ranks
/// 3. Verify last[rank i] <= first[rank i+1] for all i
/// 4. Allreduce to get global result
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
bool is_sorted([[maybe_unused]] ExecutionPolicy&& policy,
               const Container& container,
               Compare comp,
               Comm& comm) {
    using value_type = typename Container::value_type;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    // Check if local partition is sorted
    bool local_sorted = std::is_sorted(local_v.begin(), local_v.end(), comp);

    // For single rank, local sort check is sufficient
    if (num_ranks <= 1) {
        return local_sorted;
    }

    // Check boundaries between ranks
    // Each rank sends its first element to the previous rank
    // Previous rank checks: its_last <= received_first
    bool boundary_ok = true;

    if (local_size > 0) {
        // Send first element to previous rank (if not rank 0)
        // Receive first element from next rank (if not last rank)
        value_type my_first = local_v[0];
        value_type my_last = local_v[local_size - 1];
        value_type next_first{};

        if (my_rank < num_ranks - 1) {
            // Receive from next rank
            comm.recv(&next_first, sizeof(value_type), my_rank + 1, 0);
            // Check boundary: my_last should not be greater than next_first
            if (comp(next_first, my_last)) {
                boundary_ok = false;  // next_first < my_last means not sorted
            }
        }

        if (my_rank > 0) {
            // Send to previous rank
            comm.send(&my_first, sizeof(value_type), my_rank - 1, 0);
        }
    }

    // Combine local_sorted and boundary_ok
    bool my_result = local_sorted && boundary_ok;

    // Allreduce with logical AND to get global result
    return comm.allreduce_land_value(my_result);
}

/// @brief Check if local partition is sorted
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
bool local_is_sorted(const Container& container, Compare comp = Compare{}) {
    auto local_v = container.local_view();
    return std::is_sorted(local_v.begin(), local_v.end(), std::move(comp));
}

// ============================================================================
// Async sort
// ============================================================================

/// @brief Asynchronously sort distributed container
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param container The distributed container
/// @param comp Comparison function
/// @return Future indicating completion
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto async_sort(Container& container, Compare comp = Compare{})
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = sort(seq{}, container, std::move(comp));
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
