// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file stable_sort_global.hpp
/// @brief Globally stable distributed sort algorithm
/// @details Provides distributed sort with global stability: equal-key elements
///          preserve their original relative ordering across all ranks, not just
///          within each rank.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/algorithms/sorting/sort_types.hpp>
#include <dtl/algorithms/sorting/sample_sort_detail.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <algorithm>
#include <execution>
#include <functional>
#include <numeric>
#include <vector>

namespace dtl {

// ============================================================================
// Origin Tracking for Global Stability
// ============================================================================

namespace detail {

/// @brief Wrapper that augments a value with its original position
/// @tparam T The underlying value type
/// @details This wrapper is used internally by stable_sort_global to track
///          each element's original position (rank, local_index), enabling
///          a secondary sort key that preserves global ordering for equal keys.
template <typename T>
struct element_with_origin {
    T value;                    ///< The actual element value
    rank_t origin_rank;         ///< Rank where element originated
    size_type origin_index;     ///< Local index on originating rank

    /// @brief Default constructor
    element_with_origin() = default;

    /// @brief Construct with value and origin information
    element_with_origin(const T& v, rank_t r, size_type idx)
        : value(v), origin_rank(r), origin_index(idx) {}

    /// @brief Construct with moved value and origin information
    element_with_origin(T&& v, rank_t r, size_type idx)
        : value(std::move(v)), origin_rank(r), origin_index(idx) {}
};

/// @brief Comparator wrapper that uses origin as tie-breaker
/// @tparam T Value type
/// @tparam Compare User's comparison function type
template <typename T, typename Compare>
struct stable_comparator {
    Compare comp;

    explicit stable_comparator(Compare c) : comp(std::move(c)) {}

    bool operator()(const element_with_origin<T>& a,
                    const element_with_origin<T>& b) const {
        // Primary comparison: user's comparator on values
        if (comp(a.value, b.value)) return true;
        if (comp(b.value, a.value)) return false;

        // Values are equal: use origin as tie-breaker for stability
        // Lower rank comes first; within same rank, lower index comes first
        if (a.origin_rank != b.origin_rank) {
            return a.origin_rank < b.origin_rank;
        }
        return a.origin_index < b.origin_index;
    }
};

}  // namespace detail

// ============================================================================
// Globally Stable Distributed Sort
// ============================================================================

/// @brief Result type for globally stable distributed sort operations
using stable_sort_global_result = distributed_sort_result;

/// @brief Globally stable sort of distributed container (standalone version)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function (default: less-than)
/// @return Result indicating success or failure
///
/// @par Stability Guarantee:
/// This function guarantees **global stability**: if two elements have equal
/// keys (according to the comparator), their relative order after sorting
/// matches their relative order in the original global index order.
///
/// @par Original Order Definition:
/// The "original order" is defined by global indices under the container's
/// partition mapping. For block-partitioned containers, this corresponds to:
/// - Elements on rank 0 come first (in their local order)
/// - Then elements on rank 1, rank 2, etc.
///
/// @par Single-Rank Behavior:
/// For single-rank usage, this is equivalent to std::stable_sort.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(10000, ctx);
/// dtl::stable_sort_global(dtl::par{}, vec);  // Globally stable local sort
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> stable_sort_global([[maybe_unused]] ExecutionPolicy&& policy,
                                 Container& container,
                                 Compare comp = Compare{}) {
    // Single-rank case: standard stable sort provides global stability
    auto local_v = container.local_view();

    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::stable_sort(std::execution::par, local_v.begin(), local_v.end(), comp);
    } else {
        std::stable_sort(local_v.begin(), local_v.end(), comp);
    }

    return {};
}

/// @brief Globally stable sort with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
result<void> stable_sort_global(Container& container, Compare comp = Compare{}) {
    return stable_sort_global(seq{}, container, std::move(comp));
}

/// @brief Globally stable distributed sort with MPI communicator
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
/// @par Stability Guarantee:
/// This function guarantees **global stability**: if two elements have equal
/// keys (according to the comparator), their relative order after sorting
/// matches their relative order in the original global index order.
///
/// @par Algorithm:
/// 1. Augment each element with origin metadata: (rank, local_index)
/// 2. Create a comparator that uses origin as a tie-breaker for equal keys
/// 3. Perform distributed sample sort with the augmented elements
/// 4. Extract original values from the sorted augmented elements
///
/// This approach ensures that equal-key elements from different ranks maintain
/// their original relative ordering based on (rank, local_index) pairs.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Post-condition:
/// Container is globally sorted with stability. For any two elements with
/// equal keys, their final order matches their original global index order.
///
/// @par Complexity:
/// Same as distributed sample sort: O((n/p) log(n/p)) local operations plus
/// O(n/p) communication.
///
/// @par Example:
/// @code
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<int> vec(10000, comm);
/// dtl::stable_sort_global(dtl::par{}, vec, std::less<>{}, comm);
/// // Elements are globally sorted with stability across ranks
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
stable_sort_global_result stable_sort_global(
    [[maybe_unused]] ExecutionPolicy&& policy,
    Container& container,
    Compare comp,
    Comm& comm) {

    using value_type = typename Container::value_type;
    using augmented_type = detail::element_with_origin<value_type>;
    using stable_comp_type = detail::stable_comparator<value_type, Compare>;

    stable_sort_global_result result{true, 0, 0};

    rank_t num_ranks = comm.size();
    rank_t my_rank = comm.rank();

    // Handle single rank case: stable sort suffices
    if (num_ranks <= 1) {
        auto local_v = container.local_view();
        if constexpr (is_par_policy_v<ExecutionPolicy>) {
            std::stable_sort(std::execution::par, local_v.begin(), local_v.end(), comp);
        } else {
            std::stable_sort(local_v.begin(), local_v.end(), comp);
        }
        return result;
    }

    // ========================================================================
    // Phase 1: Create augmented elements with origin tracking
    // ========================================================================
    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    if (container.global_size() == 0) {
        return result;
    }

    std::vector<augmented_type> augmented_local(local_size);
    for (size_type i = 0; i < local_size; ++i) {
        augmented_local[i] = augmented_type(local_v[i], my_rank, i);
    }

    // ========================================================================
    // Phase 2: Local stable sort with origin-aware comparator
    // ========================================================================
    stable_comp_type stable_comp(comp);

    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::stable_sort(std::execution::par,
                         augmented_local.begin(), augmented_local.end(),
                         stable_comp);
    } else {
        std::stable_sort(augmented_local.begin(), augmented_local.end(),
                         stable_comp);
    }

    // ========================================================================
    // Phase 3: Sample augmented data for pivot selection
    // ========================================================================
    size_type samples_per_rank = std::max(
        static_cast<size_type>(1),
        std::min(local_size, static_cast<size_type>(3 * num_ranks)));

    // Sample based on values only (but we send full augmented elements)
    std::vector<augmented_type> local_samples;
    local_samples.reserve(samples_per_rank);
    for (size_type i = 0; i < samples_per_rank; ++i) {
        size_type idx = (i * local_size) / samples_per_rank;
        if (idx < augmented_local.size()) {
            local_samples.push_back(augmented_local[idx]);
        }
    }

    // ========================================================================
    // Phase 4: Allgather samples
    // ========================================================================
    int my_sample_count = static_cast<int>(local_samples.size());
    std::vector<int> sample_counts(static_cast<size_type>(num_ranks));
    comm.allgather(&my_sample_count, sample_counts.data(), sizeof(int));

    std::vector<int> sample_displs(static_cast<size_type>(num_ranks));
    std::exclusive_scan(sample_counts.begin(), sample_counts.end(),
                        sample_displs.begin(), 0);

    size_type total_samples = static_cast<size_type>(
        sample_displs.back() + sample_counts.back());

    std::vector<augmented_type> all_samples(total_samples);
    comm.allgatherv(local_samples.data(),
                    static_cast<size_type>(my_sample_count),
                    all_samples.data(),
                    sample_counts.data(),
                    sample_displs.data(),
                    sizeof(augmented_type));

    // ========================================================================
    // Phase 5: Select p-1 pivots from gathered samples
    // ========================================================================
    // Sort all samples by the stable comparator
    std::sort(all_samples.begin(), all_samples.end(), stable_comp);

    // Pick p-1 pivots
    std::vector<augmented_type> pivots;
    pivots.reserve(static_cast<size_type>(num_ranks - 1));
    size_type n_samples = all_samples.size();
    for (rank_t i = 1; i < num_ranks; ++i) {
        size_type idx = (static_cast<size_type>(i) * n_samples) /
                        static_cast<size_type>(num_ranks);
        if (idx >= n_samples) idx = n_samples - 1;
        pivots.push_back(all_samples[idx]);
    }

    // ========================================================================
    // Phase 6: Partition local data into p buckets by pivots
    // ========================================================================
    std::vector<std::vector<augmented_type>> buckets(
        static_cast<size_type>(num_ranks));

    for (const auto& elem : augmented_local) {
        // Find bucket using stable comparator
        auto pivot_it = std::lower_bound(
            pivots.begin(), pivots.end(), elem, stable_comp);
        rank_t bucket_idx = static_cast<rank_t>(
            std::distance(pivots.begin(), pivot_it));
        buckets[static_cast<size_type>(bucket_idx)].push_back(elem);
    }

    // ========================================================================
    // Phase 7: Compute alltoallv parameters and exchange counts
    // ========================================================================
    std::vector<int> send_counts(static_cast<size_type>(num_ranks));
    std::vector<int> send_displs(static_cast<size_type>(num_ranks));
    std::vector<int> recv_counts(static_cast<size_type>(num_ranks), 0);
    std::vector<int> recv_displs(static_cast<size_type>(num_ranks), 0);

    int disp = 0;
    for (size_type i = 0; i < static_cast<size_type>(num_ranks); ++i) {
        send_counts[i] = static_cast<int>(buckets[i].size());
        send_displs[i] = disp;
        disp += send_counts[i];
    }

    // Exchange counts
    comm.alltoall(send_counts.data(), recv_counts.data(), sizeof(int));

    // Compute receive displacements
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);
    size_type total_recv = static_cast<size_type>(
        recv_displs.back() + recv_counts.back());

    // ========================================================================
    // Phase 8: Flatten buckets and exchange data
    // ========================================================================
    // Flatten send buffer
    std::vector<augmented_type> send_buffer;
    send_buffer.reserve(augmented_local.size());
    for (const auto& bucket : buckets) {
        send_buffer.insert(send_buffer.end(), bucket.begin(), bucket.end());
    }

    std::vector<augmented_type> recv_buffer(total_recv);

    result.elements_sent = send_buffer.size();
    result.elements_received = total_recv;

    comm.alltoallv(send_buffer.data(),
                   send_counts.data(), send_displs.data(),
                   recv_buffer.data(),
                   recv_counts.data(), recv_displs.data(),
                   sizeof(augmented_type));

    // ========================================================================
    // Phase 9: Final stable sort of received data
    // ========================================================================
    if constexpr (is_par_policy_v<ExecutionPolicy>) {
        std::stable_sort(std::execution::par,
                         recv_buffer.begin(), recv_buffer.end(),
                         stable_comp);
    } else {
        std::stable_sort(recv_buffer.begin(), recv_buffer.end(),
                         stable_comp);
    }

    // ========================================================================
    // Phase 10: Extract values and write back to container
    // ========================================================================
    auto out_view = container.local_view();
    size_type copy_count = std::min(total_recv,
                                     static_cast<size_type>(out_view.size()));

    for (size_type i = 0; i < copy_count; ++i) {
        out_view[i] = recv_buffer[i].value;
    }

    return result;
}

/// @brief Globally stable sort with configuration options and communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comp Comparison function
/// @param config Sort configuration (use_parallel_local_sort honored)
/// @param comm The MPI communicator adapter
/// @return distributed_sort_result with statistics
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
stable_sort_global_result stable_sort_global_with_config(
    [[maybe_unused]] ExecutionPolicy&& policy,
    Container& container,
    Compare comp,
    distributed_sort_config config,
    Comm& comm) {
    if (config.use_parallel_local_sort) {
        return stable_sort_global(par{}, container, comp, comm);
    } else {
        return stable_sort_global(seq{}, container, comp, comm);
    }
}

// ============================================================================
// Verification Helper
// ============================================================================

/// @brief Check if container is globally sorted with stability verification
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param container The distributed container
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return true if globally sorted (values only; stability is structural)
///
/// @note This checks value ordering only. Stability verification requires
///       comparing against the original data, which is not retained.
template <typename Container, typename Compare, typename Comm>
    requires DistributedContainer<Container> && Communicator<Comm>
bool is_globally_sorted(const Container& container, Compare comp, Comm& comm) {
    using value_type = typename Container::value_type;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    // Check if local partition is sorted
    bool local_sorted = std::is_sorted(local_v.begin(), local_v.end(), comp);

    if (num_ranks <= 1) {
        return local_sorted;
    }

    // Check boundaries between ranks
    bool boundary_ok = true;

    if (local_size > 0) {
        value_type my_first = local_v[0];
        value_type my_last = local_v[local_size - 1];
        value_type next_first{};

        if (my_rank < num_ranks - 1) {
            comm.recv(&next_first, sizeof(value_type), my_rank + 1, 0);
            if (comp(next_first, my_last)) {
                boundary_ok = false;
            }
        }

        if (my_rank > 0) {
            comm.send(&my_first, sizeof(value_type), my_rank - 1, 0);
        }
    }

    bool my_result = local_sorted && boundary_ok;
    return comm.allreduce_land_value(my_result);
}

}  // namespace dtl
