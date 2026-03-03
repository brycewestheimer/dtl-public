// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file unique.hpp
/// @brief Distributed unique algorithm
/// @details Remove consecutive duplicates from distributed container.
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
#include <numeric>
#include <vector>

namespace dtl {

// ============================================================================
// Unique Result Type
// ============================================================================

/// @brief Result of distributed unique operation
struct unique_result {
    /// @brief New global size after removing duplicates
    size_type new_size = 0;

    /// @brief Number of duplicates removed
    size_type removed_count = 0;

    /// @brief Whether the operation succeeded
    bool success = true;
};

// ============================================================================
// Distributed unique
// ============================================================================

/// @brief Remove consecutive duplicates from distributed container
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Binary predicate for equality (default: equal_to)
/// @return Result containing new size and removed count
///
/// @par Precondition:
/// Container should be sorted for meaningful results.
///
/// @par Post-condition:
/// No two consecutive elements compare equal. Global size is reduced.
/// Elements after new logical end are unspecified.
///
/// @par Algorithm:
/// Performs local std::unique on this rank's partition. Boundary duplicates
/// between ranks are NOT handled by this overload. For distributed unique
/// with cross-rank boundary handling, use the overload accepting a
/// communicator: unique(policy, container, pred, comm).
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// dtl::sort(dtl::par{}, vec);  // Sort first
/// auto result = dtl::unique(dtl::par{}, vec);
/// // result.new_size is the local unique count
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate = std::equal_to<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<unique_result> unique([[maybe_unused]] ExecutionPolicy&& policy,
                             Container& container,
                             BinaryPredicate pred = BinaryPredicate{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::unique");

    unique_result res;

    auto local_view = container.local_view();
    size_type original_local_size = static_cast<size_type>(local_view.end() - local_view.begin());

    // Local unique
    auto new_end = std::unique(local_view.begin(), local_view.end(), pred);
    size_type new_local_size = static_cast<size_type>(new_end - local_view.begin());

    res.new_size = new_local_size;
    res.removed_count = original_local_size - new_local_size;

    // Note: Without a communicator, boundary duplicates between ranks cannot be
    // handled. This overload performs local unique only. For distributed unique
    // with cross-rank boundary handling, use the overload that accepts a
    // communicator parameter: unique(policy, container, pred, comm).

    return result<unique_result>{res};
}

/// @brief Unique with default execution
template <typename Container, typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("unique(container, pred) is local-only for multi-rank containers; use unique(..., pred, comm) for collective semantics or local_unique(...) for rank-local semantics")
result<unique_result> unique(Container& container,
                             BinaryPredicate pred = BinaryPredicate{}) {
    return unique(seq{}, container, std::move(pred));
}

/// @brief Distributed unique with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Binary predicate for equality
/// @param comm The MPI communicator adapter
/// @return Result containing new size and removed count
///
/// @par Algorithm:
/// 1. Local unique on each rank
/// 2. Exchange boundary elements between adjacent ranks
/// 3. If last[rank i] == first[rank i+1], mark for removal
/// 4. Remove boundary duplicates
/// 5. Compact data and return new global size
///
/// @par Precondition:
/// Container should be sorted for meaningful results.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<unique_result> unique([[maybe_unused]] ExecutionPolicy&& policy,
                             Container& container,
                             BinaryPredicate pred,
                             Comm& comm) {
    using value_type = typename Container::value_type;
    unique_result res;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    auto local_v = container.local_view();
    // Step 1: Local unique
    auto new_end = std::unique(local_v.begin(), local_v.end(), pred);
    size_type new_local_size = static_cast<size_type>(new_end - local_v.begin());
    std::vector<value_type> local_unique_values(local_v.begin(), local_v.begin() + new_local_size);

    // Step 2: Gather compacted local sizes
    int my_count = static_cast<int>(local_unique_values.size());
    std::vector<int> recv_counts(static_cast<size_type>(num_ranks), 0);
    comm.allgather(&my_count, recv_counts.data(), sizeof(int));

    std::vector<int> recv_displs(static_cast<size_type>(num_ranks), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);
    const size_type gathered_count = recv_counts.empty()
        ? 0
        : static_cast<size_type>(recv_displs.back() + recv_counts.back());

    // Step 3: Gather compacted values in global rank order
    std::vector<value_type> gathered(gathered_count);
    comm.allgatherv(local_unique_values.data(), local_unique_values.size(),
                    gathered.data(), recv_counts.data(), recv_displs.data(),
                    sizeof(value_type));

    // Step 4: Global consecutive-duplicate elimination
    std::vector<value_type> globally_unique;
    globally_unique.reserve(gathered.size());
    for (const auto& value : gathered) {
        if (globally_unique.empty() || !pred(globally_unique.back(), value)) {
            globally_unique.push_back(value);
        }
    }

    const size_type new_global_size = globally_unique.size();

    // Step 5: Repartition globally unique values back to canonical block layout
    const size_type original_global_size = container.global_size();
    const size_type base = (num_ranks > 0)
        ? (new_global_size / static_cast<size_type>(num_ranks))
        : 0;
    const size_type rem = (num_ranks > 0)
        ? (new_global_size % static_cast<size_type>(num_ranks))
        : 0;
    const size_type rank_idx = static_cast<size_type>(my_rank);
    const size_type my_target_size = base + ((rank_idx < rem) ? 1 : 0);
    const size_type my_target_offset = rank_idx * base + std::min(rank_idx, rem);

    typename Container::storage_type rebuilt_local;
    rebuilt_local.insert(rebuilt_local.end(),
                         globally_unique.begin() + static_cast<std::ptrdiff_t>(my_target_offset),
                         globally_unique.begin() + static_cast<std::ptrdiff_t>(my_target_offset + my_target_size));

    if constexpr (requires(Container& c, typename Container::storage_type data, size_type n) {
                      c.replace_local_partition(std::move(data), n);
                  }) {
        auto apply_result = container.replace_local_partition(std::move(rebuilt_local), new_global_size);
        if (!apply_result) {
            return result<unique_result>::failure(apply_result.error());
        }
    } else {
        return result<unique_result>::failure(
            status{status_code::not_supported, no_rank,
                   "Container does not support structural replacement for distributed unique"});
    }

    res.new_size = new_global_size;
    res.removed_count = original_global_size - new_global_size;

    return result<unique_result>{res};
}

// ============================================================================
// Unique copy
// ============================================================================

/// @brief Copy with consecutive duplicates removed
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam BinaryPredicate Equality predicate type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param pred Binary predicate for equality
/// @return Result containing new size
///
/// @par Note:
/// Input container is not modified.
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename BinaryPredicate = std::equal_to<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<unique_result> unique_copy(ExecutionPolicy&& policy,
                                  const InputContainer& input,
                                  OutputContainer& output,
                                  BinaryPredicate pred = BinaryPredicate{}) {
    detail::require_collective_comm_or_single_rank(input, "dtl::unique_copy");
    detail::require_collective_comm_or_single_rank(output, "dtl::unique_copy");

    unique_result res;

    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto out_end = std::unique_copy(in_local.begin(), in_local.end(),
                                     out_local.begin(), pred);
    res.new_size = static_cast<size_type>(out_end - out_local.begin());
    res.removed_count = static_cast<size_type>(in_local.end() - in_local.begin()) -
                        res.new_size;

    // Note: Without a communicator, boundary duplicates between ranks cannot be
    // handled. This overload copies with local unique only.

    return result<unique_result>{res};
}

/// @brief Unique copy with default execution
template <typename InputContainer, typename OutputContainer,
          typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
DTL_DEPRECATED_MSG("unique_copy(input, output, pred) is local-only for multi-rank containers; use communicator-aware workflows for global semantics")
result<unique_result> unique_copy(const InputContainer& input,
                                  OutputContainer& output,
                                  BinaryPredicate pred = BinaryPredicate{}) {
    return unique_copy(seq{}, input, output, std::move(pred));
}

// ============================================================================
// Local-only unique (no communication)
// ============================================================================

/// @brief Remove consecutive duplicates in local partition only
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @param container The distributed container
/// @param pred Equality predicate
/// @return Number of elements removed locally
///
/// @note NOT collective - modifies local data only. Boundary duplicates
///       between ranks are not handled.
template <typename Container, typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<Container>
size_type local_unique(Container& container,
                       BinaryPredicate pred = BinaryPredicate{}) {
    auto local_view = container.local_view();
    size_type original_size = static_cast<size_type>(local_view.end() - local_view.begin());
    auto new_end = std::unique(local_view.begin(), local_view.end(), std::move(pred));
    size_type new_size = static_cast<size_type>(new_end - local_view.begin());
    return original_size - new_size;
}

// ============================================================================
// Duplicate counting
// ============================================================================

/// @brief Count total duplicate elements globally
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @param policy Execution policy
/// @param container The distributed container (should be sorted)
/// @param pred Equality predicate
/// @return Total number of duplicate elements
///
/// @par Note:
/// Does not modify the container.
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate = std::equal_to<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
size_type count_duplicates([[maybe_unused]] ExecutionPolicy&& policy,
                           const Container& container,
                           BinaryPredicate pred = BinaryPredicate{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::count_duplicates");

    size_type local_dups = 0;

    auto local_view = container.local_view();
    if (local_view.begin() == local_view.end()) {
        return 0;
    }

    auto prev = local_view.begin();
    for (auto it = prev + 1; it != local_view.end(); ++it) {
        if (pred(*prev, *it)) {
            ++local_dups;
        }
        prev = it;
    }

    // Note: Without a communicator, boundary duplicates and global reduction
    // cannot be performed. Use count_duplicates(policy, container, pred, comm)
    // for a globally correct count.
    return local_dups;
}

/// @brief Count duplicates with default execution
template <typename Container, typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("count_duplicates(container, pred) is local-only for multi-rank containers; use count_duplicates(..., pred, comm) for global semantics")
size_type count_duplicates(const Container& container,
                           BinaryPredicate pred = BinaryPredicate{}) {
    return count_duplicates(seq{}, container, std::move(pred));
}

/// @brief Count total duplicate elements globally with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Equality predicate
/// @param comm The MPI communicator adapter
/// @return Total number of duplicate elements globally
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
size_type count_duplicates([[maybe_unused]] ExecutionPolicy&& policy,
                           const Container& container,
                           BinaryPredicate pred,
                           Comm& comm) {
    using value_type = typename Container::value_type;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    // Count local duplicates
    long local_dups = 0;
    if (local_size > 1) {
        auto prev = local_v.begin();
        for (auto it = prev + 1; it != local_v.end(); ++it) {
            if (pred(*prev, *it)) {
                ++local_dups;
            }
            prev = it;
        }
    }

    // Check boundary duplicates
    if (num_ranks > 1 && local_size > 0) {
        value_type my_first = local_v[0];
        value_type my_last = local_v[local_size - 1];

        if (my_rank > 0) {
            value_type prev_last{};
            comm.recv(&prev_last, sizeof(value_type), my_rank - 1, 0);
            if (pred(prev_last, my_first)) {
                ++local_dups;
            }
        }

        if (my_rank < num_ranks - 1) {
            comm.send(&my_last, sizeof(value_type), my_rank + 1, 0);
        }
    } else if (num_ranks > 1) {
        // Empty partition: participate in communication
        value_type dummy{};
        if (my_rank > 0) {
            comm.recv(&dummy, sizeof(value_type), my_rank - 1, 0);
        }
        if (my_rank < num_ranks - 1) {
            comm.send(&dummy, sizeof(value_type), my_rank + 1, 0);
        }
    }

    // Allreduce to get global count
    long global_dups = comm.template allreduce_sum_value<long>(local_dups);
    return static_cast<size_type>(global_dups);
}

// ============================================================================
// Has duplicates check
// ============================================================================

/// @brief Check if container has any consecutive duplicates
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Equality predicate
/// @return true if any consecutive duplicates exist
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate = std::equal_to<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
bool has_duplicates([[maybe_unused]] ExecutionPolicy&& policy,
                    const Container& container,
                    BinaryPredicate pred = BinaryPredicate{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::has_duplicates");

    auto local_view = container.local_view();
    auto it = std::adjacent_find(local_view.begin(), local_view.end(), pred);
    bool local_has = (it != local_view.end());

    // Note: Without a communicator, boundary duplicates and global reduction
    // cannot be checked. Use has_duplicates(policy, container, pred, comm)
    // for a globally correct result.
    return local_has;
}

/// @brief Check for duplicates with default execution
template <typename Container, typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("has_duplicates(container, pred) is local-only for multi-rank containers; use has_duplicates(..., pred, comm) for global semantics")
bool has_duplicates(const Container& container,
                    BinaryPredicate pred = BinaryPredicate{}) {
    return has_duplicates(seq{}, container, std::move(pred));
}

/// @brief Check if container has any consecutive duplicates with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam BinaryPredicate Equality predicate type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param pred Equality predicate
/// @param comm The MPI communicator adapter
/// @return true if any consecutive duplicates exist globally
template <typename ExecutionPolicy, typename Container,
          typename BinaryPredicate, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
bool has_duplicates([[maybe_unused]] ExecutionPolicy&& policy,
                    const Container& container,
                    BinaryPredicate pred,
                    Comm& comm) {
    using value_type = typename Container::value_type;

    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    // Check local duplicates
    auto it = std::adjacent_find(local_v.begin(), local_v.end(), pred);
    bool local_has = (it != local_v.end());

    // Check boundary duplicates
    if (num_ranks > 1 && local_size > 0) {
        value_type my_first = local_v[0];
        value_type my_last = local_v[local_size - 1];

        if (my_rank > 0) {
            value_type prev_last{};
            comm.recv(&prev_last, sizeof(value_type), my_rank - 1, 0);
            if (pred(prev_last, my_first)) {
                local_has = true;
            }
        }

        if (my_rank < num_ranks - 1) {
            comm.send(&my_last, sizeof(value_type), my_rank + 1, 0);
        }
    } else if (num_ranks > 1) {
        // Empty partition: participate in communication
        value_type dummy{};
        if (my_rank > 0) {
            comm.recv(&dummy, sizeof(value_type), my_rank - 1, 0);
        }
        if (my_rank < num_ranks - 1) {
            comm.send(&dummy, sizeof(value_type), my_rank + 1, 0);
        }
    }

    // Allreduce with logical OR
    return comm.allreduce_lor_value(local_has);
}

// ============================================================================
// Async unique
// ============================================================================

/// @brief Asynchronously remove duplicates
template <typename Container, typename BinaryPredicate = std::equal_to<>>
    requires DistributedContainer<Container>
auto async_unique(Container& container,
                  BinaryPredicate pred = BinaryPredicate{})
    -> result<unique_result> {
    return unique(async{}, container, std::move(pred));
}

}  // namespace dtl
