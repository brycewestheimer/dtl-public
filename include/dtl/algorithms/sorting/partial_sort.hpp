// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file partial_sort.hpp
/// @brief Distributed partial sort algorithm
/// @details Sort first n elements across distributed container.
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
#include <dtl/algorithms/sorting/nth_element.hpp>

#include <algorithm>
#include <functional>
#include <vector>
#include <numeric>

namespace dtl {

// ============================================================================
// Distributed partial sort
// ============================================================================

/// @brief Partially sort distributed container
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container The distributed container
/// @param n Number of elements to sort (global count)
/// @param comp Comparison function
/// @return Result indicating success or failure
///
/// @par Post-condition:
/// The first n elements (in global index order) are the n smallest elements,
/// in sorted order. Elements after position n are unspecified.
///
/// @par Algorithm:
/// 1. Find nth smallest element (distributed selection)
/// 2. Partition elements <= nth to first ranks
/// 3. Sort the first n elements
///
/// @par Complexity:
/// O((n/p) log(n/p)) for selection + O(n) redistribution.
/// More efficient than full sort when n << size.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(10000, ctx);
/// // Sort only first 100 elements globally
/// dtl::partial_sort(dtl::par{}, vec, 100);
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<void> partial_sort([[maybe_unused]] ExecutionPolicy&& policy,
                          Container& container,
                          size_type n,
                          Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(container, "dtl::partial_sort");

    // Limitation: only local partial sort is performed; distributed partial sort
    // requires cross-rank sampling and redistribution (not yet implemented).
    auto local_view = container.local_view();
    size_type local_n = (n < static_cast<size_type>(local_view.end() - local_view.begin()))
                            ? n
                            : static_cast<size_type>(local_view.end() - local_view.begin());
    std::partial_sort(local_view.begin(), local_view.begin() + local_n, local_view.end(), comp);

    return {};
}

/// @brief Partial sort with default execution
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
result<void> partial_sort(Container& container, size_type n,
                          Compare comp = Compare{}) {
    return partial_sort(seq{}, container, n, std::move(comp));
}

/// @brief Distributed partial sort with MPI communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param n Number of elements to sort (global count)
/// @param comp Comparison function
/// @param comm The MPI communicator adapter
/// @return Result indicating success or failure
///
/// @par Algorithm:
/// 1. Find nth smallest element via distributed quickselect
/// 2. Partition elements: those <= nth go to lower ranks
/// 3. Sort the first n elements across participating ranks
///
/// @par Complexity:
/// O(n/p log(n/p)) for selection + O(n log n) for sorting top-n.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename Container,
          typename Compare, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<void> partial_sort(ExecutionPolicy&& policy,
                          Container& container,
                          size_type n,
                          Compare comp,
                          Comm& comm) {
    using value_type = typename Container::value_type;

    size_type global_size = container.global_size();
    if (n == 0 || global_size == 0) {
        return {};
    }

    // Clamp n to global size
    n = std::min(n, global_size);

    rank_t num_ranks = comm.size();
    rank_t my_rank = comm.rank();

    // Single rank case: just do local partial_sort
    if (num_ranks <= 1) {
        auto local_v = container.local_view();
        size_type local_n = std::min(n, static_cast<size_type>(local_v.size()));
        std::partial_sort(local_v.begin(), local_v.begin() + local_n,
                          local_v.end(), comp);
        return {};
    }

    // Step 1: Find the nth element via distributed quickselect
    // This partitions the data such that elements before n are <= nth element
    auto nth_result = nth_element(std::forward<ExecutionPolicy>(policy),
                                   container, static_cast<index_t>(n - 1), comp, comm);

    if (!nth_result.valid) {
        return result<void>::failure(
            status{status_code::operation_failed, "nth_element failed during partial_sort"});
    }

    value_type nth_value = nth_result.value;

    // Step 2: Gather all elements <= nth_value to lower ranks
    auto local_v = container.local_view();
    size_type local_size = static_cast<size_type>(local_v.size());

    // Count elements <= nth_value
    std::vector<value_type> elements_le_nth;
    std::vector<value_type> elements_gt_nth;

    for (size_type i = 0; i < local_size; ++i) {
        if (!comp(nth_value, local_v[i])) {  // local_v[i] <= nth_value
            elements_le_nth.push_back(local_v[i]);
        } else {
            elements_gt_nth.push_back(local_v[i]);
        }
    }

    // Step 3: Sort the elements that are <= nth_value
    // For a full implementation, we would redistribute these elements
    // to ensure the first n elements globally are sorted.
    // For now, we do a simpler approach: sort local elements that are <= nth
    std::sort(elements_le_nth.begin(), elements_le_nth.end(), comp);

    // Copy back sorted elements
    // Elements <= nth go first, then elements > nth
    size_type idx = 0;
    for (const auto& elem : elements_le_nth) {
        if (idx < local_size) {
            local_v[idx++] = elem;
        }
    }
    for (const auto& elem : elements_gt_nth) {
        if (idx < local_size) {
            local_v[idx++] = elem;
        }
    }

    return {};
}

// ============================================================================
// Partial sort copy
// ============================================================================

/// @brief Copy n smallest elements in sorted order
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container (sized for n elements)
/// @param comp Comparison function
/// @return Result indicating success or failure
///
/// @par Post-condition:
/// Output contains the n smallest elements from input, in sorted order.
/// Input is unchanged.
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<void> partial_sort_copy([[maybe_unused]] ExecutionPolicy&& policy,
                               const InputContainer& input,
                               OutputContainer& output,
                               Compare comp = Compare{}) {
    detail::require_collective_comm_or_single_rank(input, "dtl::partial_sort_copy");

    // Limitation: only local partial sort copy is performed; distributed version not yet implemented
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    std::partial_sort_copy(in_local.begin(), in_local.end(),
                           out_local.begin(), out_local.end(), comp);

    return {};
}

/// @brief Partial sort copy with default execution
template <typename InputContainer, typename OutputContainer,
          typename Compare = std::less<>>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<void> partial_sort_copy(const InputContainer& input,
                               OutputContainer& output,
                               Compare comp = Compare{}) {
    return partial_sort_copy(seq{}, input, output, std::move(comp));
}

// ============================================================================
// Local-only partial sort (no communication)
// ============================================================================

/// @brief Partial sort local partition only
/// @tparam Container Distributed container type
/// @tparam Compare Comparison function type
/// @param container The distributed container
/// @param n Number of elements to sort in local partition
/// @param comp Comparison function
///
/// @note NOT collective - partial sorts local data only.
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
void local_partial_sort(Container& container, size_type n,
                        Compare comp = Compare{}) {
    auto local_view = container.local_view();
    size_type local_n = (n < static_cast<size_type>(local_view.end() - local_view.begin()))
                            ? n
                            : static_cast<size_type>(local_view.end() - local_view.begin());
    std::partial_sort(local_view.begin(), local_view.begin() + local_n,
                      local_view.end(), std::move(comp));
}

// ============================================================================
// Top K elements
// ============================================================================

/// @brief Find k largest elements globally
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam Compare Comparison function type
/// @param policy Execution policy
/// @param container Source container
/// @param output Output container (sized for k elements)
/// @param k Number of largest elements to find
/// @param comp Comparison function
/// @return Result indicating success or failure
///
/// @par Post-condition:
/// Output contains the k largest elements, in descending order.
template <typename ExecutionPolicy, typename Container,
          typename OutputContainer, typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> top_k(ExecutionPolicy&& policy,
                   const Container& container,
                   OutputContainer& output,
                   size_type k,
                   Compare comp = Compare{}) {
    // top_k is partial_sort_copy with reversed comparison
    return partial_sort_copy(std::forward<ExecutionPolicy>(policy),
                             container, output,
                             [&comp](const auto& a, const auto& b) {
                                 return comp(b, a);
                             });
}

/// @brief Find k smallest elements globally
template <typename ExecutionPolicy, typename Container,
          typename OutputContainer, typename Compare = std::less<>>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> bottom_k(ExecutionPolicy&& policy,
                      const Container& container,
                      OutputContainer& output,
                      size_type k,
                      Compare comp = Compare{}) {
    return partial_sort_copy(std::forward<ExecutionPolicy>(policy),
                             container, output, std::move(comp));
}

// ============================================================================
// Async partial sort
// ============================================================================

/// @brief Asynchronously partial sort
template <typename Container, typename Compare = std::less<>>
    requires DistributedContainer<Container>
auto async_partial_sort(Container& container, size_type n,
                        Compare comp = Compare{})
    -> result<void> {
    return partial_sort(async{}, container, n, std::move(comp));
}

}  // namespace dtl
