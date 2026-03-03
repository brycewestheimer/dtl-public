// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file adjacent_difference.hpp
/// @brief Distributed adjacent_difference algorithm
/// @details Compute differences between consecutive elements across distributed containers.
///          For distributed mode, boundary elements are communicated between adjacent ranks.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>

namespace dtl {

// ============================================================================
// Adjacent Difference Result Type
// ============================================================================

/// @brief Result of a distributed adjacent_difference operation
struct adjacent_difference_result {
    /// @brief Number of elements written
    size_type count = 0;

    /// @brief Whether the operation completed successfully
    bool success = true;
};

// ============================================================================
// Local Adjacent Difference (no communication)
// ============================================================================

/// @brief Compute adjacent differences on local partition only
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container (must have same local_size)
/// @return Result containing count of elements written
///
/// @par Complexity:
/// O(n/p) local operations. No communication required.
///
/// @note NOT collective - computes differences within local data only.
///       The first element of each rank's local output is a copy of its first input element.
///       For cross-rank boundary correctness, use the communicator overload.
template <typename ExecutionPolicy, typename InputContainer, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<adjacent_difference_result> adjacent_difference(
    [[maybe_unused]] ExecutionPolicy&& policy,
    const InputContainer& input,
    OutputContainer& output) {
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    if (in_local.size() == 0) {
        return result<adjacent_difference_result>{{0, true}};
    }

    std::adjacent_difference(in_local.begin(), in_local.end(), out_local.begin());

    return result<adjacent_difference_result>{
        {static_cast<size_type>(in_local.size()), true}};
}

/// @brief Adjacent difference with default execution
template <typename InputContainer, typename OutputContainer>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<adjacent_difference_result> adjacent_difference(
    const InputContainer& input,
    OutputContainer& output) {
    return adjacent_difference(seq{}, input, output);
}

// ============================================================================
// Adjacent Difference with Custom Binary Operation
// ============================================================================

/// @brief Compute adjacent differences using a custom binary operation
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param op Binary operation applied as op(input[i], input[i-1])
/// @return Result containing count of elements written
///
/// @par Example:
/// @code
/// dtl::adjacent_difference(dtl::seq{}, vec_in, vec_out,
///     [](int a, int b) { return a + b; });
/// @endcode
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<adjacent_difference_result> adjacent_difference(
    [[maybe_unused]] ExecutionPolicy&& policy,
    const InputContainer& input,
    OutputContainer& output,
    BinaryOp op) {
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    if (in_local.size() == 0) {
        return result<adjacent_difference_result>{{0, true}};
    }

    std::adjacent_difference(in_local.begin(), in_local.end(),
                             out_local.begin(), std::move(op));

    return result<adjacent_difference_result>{
        {static_cast<size_type>(in_local.size()), true}};
}

/// @brief Adjacent difference with custom op and default execution
template <typename InputContainer, typename OutputContainer, typename BinaryOp>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<adjacent_difference_result> adjacent_difference(
    const InputContainer& input,
    OutputContainer& output,
    BinaryOp op) {
    return adjacent_difference(seq{}, input, output, std::move(op));
}

// ============================================================================
// Adjacent Difference with Communicator (distributed boundary exchange)
// ============================================================================

/// @brief Compute adjacent differences with cross-rank boundary communication
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param comm Communicator for boundary element exchange
/// @return Result containing local count of elements written
///
/// @par Algorithm:
/// 1. Each rank (r > 0) receives the last element from rank (r-1)
/// 2. Local adjacent_difference is computed
/// 3. For rank r > 0, output[0] = input[0] - boundary_from_prev_rank
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<adjacent_difference_result> adjacent_difference(
    ExecutionPolicy&& policy,
    const InputContainer& input,
    OutputContainer& output,
    Comm& comm) {
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    if (in_local.size() == 0) {
        return result<adjacent_difference_result>{{0, true}};
    }

    // Step 1: Compute local adjacent difference
    std::adjacent_difference(in_local.begin(), in_local.end(), out_local.begin());

    // Step 2: Fix up boundary element for rank > 0
    // Each rank sends its last element to the next rank
    using value_type = typename InputContainer::value_type;
    rank_t my_rank = comm.rank();
    rank_t num_ranks = comm.size();

    if (num_ranks > 1 && in_local.size() > 0) {
        value_type boundary_value{};

        // Send last element to next rank, receive from previous rank
        if (my_rank > 0 && my_rank < num_ranks) {
            // Receive boundary from previous rank
            boundary_value = comm.template recv_value<value_type>(my_rank - 1, 0);
            // Fix up first element: output[0] = input[0] - boundary_from_prev
            out_local[0] = in_local[0] - boundary_value;
        }

        if (my_rank < num_ranks - 1) {
            // Send our last element to next rank
            value_type last = in_local[in_local.size() - 1];
            comm.template send_value<value_type>(last, my_rank + 1, 0);
        }
    }

    return result<adjacent_difference_result>{
        {static_cast<size_type>(in_local.size()), true}};
}

// ============================================================================
// Async Adjacent Difference
// ============================================================================

/// @brief Asynchronously compute adjacent differences
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param input Source container
/// @param output Destination container
/// @return Future containing result
template <typename InputContainer, typename OutputContainer>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
auto async_adjacent_difference(const InputContainer& input, OutputContainer& output)
    -> futures::distributed_future<adjacent_difference_result> {
    auto promise = std::make_shared<futures::distributed_promise<adjacent_difference_result>>();
    auto future = promise->get_future();

    try {
        auto res = adjacent_difference(seq{}, input, output);
        if (res) {
            promise->set_value(res.value());
        } else {
            promise->set_error(res.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
