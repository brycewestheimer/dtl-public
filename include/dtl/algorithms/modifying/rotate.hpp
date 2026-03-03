// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rotate.hpp
/// @brief Distributed rotate/shift algorithm
/// @details Local rotation within each rank's partition and global rotation
///          across ranks using alltoallv communication.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

namespace dtl {

// ============================================================================
// Local Rotate (within rank's partition)
// ============================================================================

/// @brief Rotate elements within local partition
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container Container to rotate
/// @param n_positions Number of positions to rotate left.
///        Positive values rotate left, negative values rotate right.
/// @return Result indicating success or failure
///
/// @par Complexity:
/// O(n/p) local element moves. No communication required.
///
/// @note This rotates ONLY within each rank's local partition.
///       Elements do not cross rank boundaries.
///       Limitation: full distributed rotation requires alltoallv redistribution.
///       Current implementation only rotates the local partition.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(100, ctx);
/// // Rank 0 local: [0, 1, 2, ..., 24]
/// dtl::local_rotate(dtl::seq{}, vec, 3);
/// // Rank 0 local: [3, 4, ..., 24, 0, 1, 2]
/// @endcode
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> local_rotate([[maybe_unused]] ExecutionPolicy&& policy,
                          Container& container,
                          long long n_positions) {
    auto local_v = container.local_view();
    if (local_v.size() == 0) {
        return {};
    }

    // Normalize rotation amount to [0, size)
    long long sz = static_cast<long long>(local_v.size());
    long long normalized = ((n_positions % sz) + sz) % sz;

    if (normalized == 0) {
        return {};
    }

    std::rotate(local_v.begin(),
                local_v.begin() + static_cast<ptrdiff_t>(normalized),
                local_v.end());
    return {};
}

/// @brief Local rotate with default execution
template <typename Container>
    requires DistributedContainer<Container>
result<void> local_rotate(Container& container, long long n_positions) {
    return local_rotate(seq{}, container, n_positions);
}

// ============================================================================
// Local Shift Left
// ============================================================================

/// @brief Shift elements left within local partition
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type for fill
/// @param policy Execution policy
/// @param container Container to shift
/// @param n Number of positions to shift left
/// @param fill_value Value to fill vacated positions at the end
/// @return Number of elements that remained (local_size - n, or 0 if n >= size)
///
/// @note Shifts within local partition only. Vacated positions at the end
///       are filled with fill_value.
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type local_shift_left([[maybe_unused]] ExecutionPolicy&& policy,
                           Container& container,
                           size_type n,
                           const T& fill_value) {
    auto local_v = container.local_view();
    if (n >= local_v.size()) {
        // Shift by more than size: fill everything
        std::fill(local_v.begin(), local_v.end(), fill_value);
        return 0;
    }

    // Shift elements left
    std::copy(local_v.begin() + n, local_v.end(), local_v.begin());
    // Fill vacated positions
    std::fill(local_v.end() - n, local_v.end(), fill_value);
    return local_v.size() - n;
}

/// @brief Shift left with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_shift_left(Container& container, size_type n, const T& fill_value) {
    return local_shift_left(seq{}, container, n, fill_value);
}

// ============================================================================
// Local Shift Right
// ============================================================================

/// @brief Shift elements right within local partition
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type for fill
/// @param policy Execution policy
/// @param container Container to shift
/// @param n Number of positions to shift right
/// @param fill_value Value to fill vacated positions at the beginning
/// @return Number of elements that remained (local_size - n, or 0 if n >= size)
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type local_shift_right([[maybe_unused]] ExecutionPolicy&& policy,
                            Container& container,
                            size_type n,
                            const T& fill_value) {
    auto local_v = container.local_view();
    if (n >= local_v.size()) {
        std::fill(local_v.begin(), local_v.end(), fill_value);
        return 0;
    }

    // Shift elements right (copy backward to avoid overwrite)
    std::copy_backward(local_v.begin(), local_v.end() - n, local_v.end());
    // Fill vacated positions at the beginning
    std::fill(local_v.begin(), local_v.begin() + n, fill_value);
    return local_v.size() - n;
}

/// @brief Shift right with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_shift_right(Container& container, size_type n, const T& fill_value) {
    return local_shift_right(seq{}, container, n, fill_value);
}

// ============================================================================
// Distributed Rotate (cross-rank rotation via alltoallv)
// ============================================================================

/// @brief Rotate elements globally across all ranks
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container Container to rotate
/// @param n_positions Number of positions to rotate left globally.
///        Positive values rotate left, negative values rotate right.
/// @param comm Communicator for inter-rank data movement
/// @return Result indicating success or failure
///
/// @par Algorithm:
/// 1. Gather local sizes from all ranks to determine global layout
/// 2. Compute effective rotation modulo global size
/// 3. For each local element, determine its destination rank after rotation
/// 4. Build send/recv counts and exchange via alltoallv
/// 5. Place received elements at correct local positions
///
/// @par Complexity:
/// O(n/p) local work + O(n/p) alltoallv communication.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<int> vec(100, comm);
/// // Global: [0, 1, 2, ..., 99]
/// dtl::rotate(dtl::seq{}, vec, 10, comm);
/// // Global: [10, 11, ..., 99, 0, 1, ..., 9]
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<void> rotate([[maybe_unused]] ExecutionPolicy&& policy,
                    Container& container,
                    long long n_positions,
                    Comm& comm) {
    using value_type = typename Container::value_type;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    // Single rank: just do local rotate
    if (num_ranks <= 1) {
        return local_rotate(seq{}, container, n_positions);
    }

    // Step 1: Gather local sizes to determine global layout
    int my_local_size = static_cast<int>(container.local_size());
    std::vector<int> local_sizes(static_cast<size_type>(num_ranks));
    comm.allgather(&my_local_size, local_sizes.data(), sizeof(int));

    // Compute prefix sums (global offset for each rank)
    std::vector<long long> prefix(static_cast<size_type>(num_ranks + 1), 0);
    for (rank_t r = 0; r < num_ranks; ++r) {
        prefix[static_cast<size_type>(r + 1)] =
            prefix[static_cast<size_type>(r)] + local_sizes[static_cast<size_type>(r)];
    }
    long long global_size = prefix[static_cast<size_type>(num_ranks)];

    if (global_size == 0) return {};

    // Step 2: Normalize rotation to [0, global_size)
    long long k = ((n_positions % global_size) + global_size) % global_size;
    if (k == 0) return {};

    long long my_lo = prefix[static_cast<size_type>(my_rank)];

    // Helper: find rank owning global position g using prefix sums
    auto rank_of = [&prefix, num_ranks](long long g) -> rank_t {
        // Upper bound gives first prefix > g, so rank = (upper_bound index) - 1
        auto it = std::upper_bound(
            prefix.begin(), prefix.begin() + num_ranks + 1, g);
        rank_t r = static_cast<rank_t>(std::distance(prefix.begin(), it)) - 1;
        if (r < 0) r = 0;
        if (r >= num_ranks) r = num_ranks - 1;
        return r;
    };

    // Step 3: Build send_counts (where my elements go after rotation)
    // Element at global position g moves to position (g - k + N) % N
    std::vector<int> send_counts(static_cast<size_type>(num_ranks), 0);

    for (int i = 0; i < my_local_size; ++i) {
        long long old_g = my_lo + i;
        long long new_g = (old_g - k + global_size) % global_size;
        rank_t dest = rank_of(new_g);
        send_counts[static_cast<size_type>(dest)]++;
    }

    // Exchange send_counts to get recv_counts
    std::vector<int> recv_counts(static_cast<size_type>(num_ranks), 0);
    comm.alltoall(send_counts.data(), recv_counts.data(), sizeof(int));

    // Compute displacements
    std::vector<int> send_displs(static_cast<size_type>(num_ranks), 0);
    std::vector<int> recv_displs(static_cast<size_type>(num_ranks), 0);
    std::exclusive_scan(send_counts.begin(), send_counts.end(),
                        send_displs.begin(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);

    int total_recv = (recv_displs.empty() ? 0 :
                      recv_displs.back() + recv_counts.back());

    // Step 4: Build send buffer (elements grouped by destination rank)
    auto local_v = container.local_view();
    std::vector<value_type> send_buffer(static_cast<size_type>(my_local_size));
    std::vector<int> bucket_pos(static_cast<size_type>(num_ranks));
    for (rank_t r = 0; r < num_ranks; ++r) {
        bucket_pos[static_cast<size_type>(r)] =
            send_displs[static_cast<size_type>(r)];
    }

    for (int i = 0; i < my_local_size; ++i) {
        long long old_g = my_lo + i;
        long long new_g = (old_g - k + global_size) % global_size;
        rank_t dest = rank_of(new_g);
        send_buffer[static_cast<size_type>(
            bucket_pos[static_cast<size_type>(dest)]++)] = local_v[static_cast<size_type>(i)];
    }

    // Step 5: Exchange data via alltoallv
    std::vector<value_type> recv_buffer(static_cast<size_type>(total_recv));
    comm.alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                   recv_buffer.data(), recv_counts.data(), recv_displs.data(),
                   sizeof(value_type));

    if constexpr (requires(Container& c, typename Container::storage_type data) {
                      c.replace_local_partition(std::move(data));
                  }) {
        typename Container::storage_type rotated_local(recv_buffer.begin(), recv_buffer.end());
        auto apply_result = container.replace_local_partition(std::move(rotated_local));
        if (!apply_result) {
            return result<void>::failure(apply_result.error());
        }
        return {};
    } else {
        return result<void>::failure(
            status{status_code::not_supported, no_rank,
                   "Container does not support structural replacement for distributed rotate"});
    }
}

// ============================================================================
// Async Local Rotate
// ============================================================================

/// @brief Asynchronously rotate local partition
/// @tparam Container Distributed container type
/// @param container Container to rotate
/// @param n_positions Number of positions to rotate left
/// @return Future indicating completion
template <typename Container>
    requires DistributedContainer<Container>
auto async_local_rotate(Container& container, long long n_positions)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto res = local_rotate(seq{}, container, n_positions);
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
