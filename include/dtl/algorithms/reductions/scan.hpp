// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file scan.hpp
/// @brief Distributed prefix scan algorithms
/// @details Implements inclusive_scan, exclusive_scan, and transform variants.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/detail/determinism_guard.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/algorithms/concepts.hpp>

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <functional>
#include <numeric>
#include <type_traits>
#include <vector>

namespace dtl {

// ============================================================================
// Distributed Inclusive Scan
// ============================================================================

/// @brief Compute inclusive prefix scan across all ranks
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @return Result indicating success or error
///
/// @par Algorithm:
/// 1. Compute local inclusive scan
/// 2. Gather last local value from each rank
/// 3. Compute prefix of last values via MPI_Scan
/// 4. Offset local results by prefix
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Example:
/// @code
/// distributed_vector<int> input(1000, ctx);
/// distributed_vector<int> output(1000, ctx);
/// dtl::inclusive_scan(dtl::par{}, input, output, 0, std::plus<>{});
/// // output[i] = input[0] + input[1] + ... + input[i]
/// @endcode
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> inclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op) {
    (void)policy;

    // Get local views
    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};  // Nothing to do
    }

    // Step 1: Compute local inclusive scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        running = binary_op(running, local_in[i]);
        local_out[i] = running;
    }

    // Step 2: For single-rank case, we're done
    if (input.num_ranks() <= 1) {
        return {};
    }

    // Step 3: Multi-rank requires explicit communicator.
    return make_error<void>(
        status_code::precondition_failed,
        "inclusive_scan requires an explicit communicator when num_ranks()>1; "
        "use global_inclusive_scan(..., comm) or local_inclusive_scan(...) for local semantics.");
}

/// @brief Inclusive scan with communicator for cross-rank prefix
/// @details Uses MPI_Exscan to compute the exclusive prefix of local sums
///          from prior ranks, then offsets all local results.
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<void> inclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op,
                            Comm& comm) {
    (void)policy;

    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::inclusive_scan");
        !deterministic_guard) {
        return deterministic_guard;
    }

    // Verify the operation is supported for distributed scan.
    // Only std::plus is supported because MPI_Exscan is used for the cross-rank prefix.
    static_assert(std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>,
                  "inclusive_scan: Only std::plus<> is supported for distributed scan. "
                  "For non-plus operations, use local_inclusive_scan() or implement "
                  "a custom cross-rank prefix exchange.");

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Step 1: Compute local inclusive scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        running = binary_op(running, local_in[i]);
        local_out[i] = running;
    }

    // Step 2: For single-rank case, we're done
    if (comm.size() <= 1) {
        return {};
    }

    // Step 3: Get exclusive prefix of local sums from prior ranks
    T local_sum = running;
    T prefix = comm.template exscan_sum_value<T>(local_sum);
    // exscan_sum_value returns T{} for rank 0

    // Step 4: Offset all local results by the prefix from prior ranks
    if (comm.rank() > 0) {
        for (size_type i = 0; i < local_out.size(); ++i) {
            local_out[i] = binary_op(prefix, local_out[i]);
        }
    }

    return {};
}

/// @brief Inclusive scan with default execution
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> inclusive_scan(const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op) {
    return inclusive_scan(seq{}, input, output, init, std::move(binary_op));
}

/// @brief Inclusive scan with plus operation
template <typename ExecutionPolicy, typename Container, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> inclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output) {
    using value_type = typename Container::value_type;
    return inclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, value_type{}, std::plus<>{});
}

// ============================================================================
// Distributed Exclusive Scan
// ============================================================================

/// @brief Compute exclusive prefix scan across all ranks
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @return Result indicating success or error
///
/// @par Algorithm:
/// output[i] = init op input[0] op input[1] op ... op input[i-1]
/// (i.e., the result does NOT include input[i])
///
/// @par Example:
/// @code
/// distributed_vector<int> input(1000, ctx);
/// distributed_vector<int> output(1000, ctx);
/// dtl::exclusive_scan(dtl::par{}, input, output, 0, std::plus<>{});
/// // output[0] = 0
/// // output[1] = input[0]
/// // output[2] = input[0] + input[1]
/// // etc.
/// @endcode
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> exclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op) {
    (void)policy;

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Step 1: Compute local exclusive scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_out[i] = running;
        running = binary_op(running, local_in[i]);
    }

    // Step 2: For single-rank case, we're done
    if (input.num_ranks() <= 1) {
        return {};
    }

    // Step 3: Multi-rank requires explicit communicator.
    return make_error<void>(
        status_code::precondition_failed,
        "exclusive_scan requires an explicit communicator when num_ranks()>1; "
        "use global_exclusive_scan(..., comm) or local_exclusive_scan(...) for local semantics.");
}

/// @brief Exclusive scan with communicator for cross-rank prefix
/// @details Uses MPI_Exscan to compute the exclusive prefix of local sums
///          from prior ranks, then offsets all local results.
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<void> exclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op,
                            Comm& comm) {
    (void)policy;

    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::exclusive_scan");
        !deterministic_guard) {
        return deterministic_guard;
    }

    // Verify the operation is supported for distributed scan.
    // Only std::plus is supported because MPI_Exscan is used for the cross-rank prefix.
    static_assert(std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>,
                  "exclusive_scan: Only std::plus<> is supported for distributed scan. "
                  "For non-plus operations, use local_exclusive_scan() or implement "
                  "a custom cross-rank prefix exchange.");

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Step 1: Compute local exclusive scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_out[i] = running;
        running = binary_op(running, local_in[i]);
    }

    // Step 2: For single-rank case, we're done
    if (comm.size() <= 1) {
        return {};
    }

    // Step 3: Get exclusive prefix of local sums from prior ranks
    // running now contains init op all_local_elements
    // local contribution is running (which is init + local sum for plus)
    T local_sum = running;
    T prefix = comm.template exscan_sum_value<T>(local_sum);
    // exscan_sum_value returns T{} for rank 0

    // Step 4: Offset all local results by the prefix from prior ranks
    if (comm.rank() > 0) {
        for (size_type i = 0; i < local_out.size(); ++i) {
            local_out[i] = binary_op(prefix, local_out[i]);
        }
    }

    return {};
}

/// @brief Exclusive scan with default execution
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> exclusive_scan(const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op) {
    return exclusive_scan(seq{}, input, output, init, std::move(binary_op));
}

/// @brief Exclusive scan with plus operation
template <typename ExecutionPolicy, typename Container, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> exclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output) {
    using value_type = typename Container::value_type;
    return exclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, value_type{}, std::plus<>{});
}

// ============================================================================
// Transform Inclusive Scan
// ============================================================================

/// @brief Transform then compute inclusive scan
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @tparam UnaryOp Transform operation type
/// @param policy Execution policy
/// @param input Input container
/// @param output Output container
/// @param init Initial value
/// @param binary_op Reduction operation
/// @param unary_op Transform operation
/// @return Result indicating success or error
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> transform_inclusive_scan(ExecutionPolicy&& policy,
                                      const Container& input,
                                      OutputContainer& output,
                                      T init,
                                      BinaryOp binary_op,
                                      UnaryOp unary_op) {
    (void)policy;

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Compute local transform-inclusive-scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        running = binary_op(running, unary_op(local_in[i]));
        local_out[i] = running;
    }

    // Step 2: For single-rank case, we're done
    if (input.num_ranks() <= 1) {
        return {};
    }

    // Step 3: Multi-rank requires explicit communicator.
    return make_error<void>(
        status_code::precondition_failed,
        "transform_inclusive_scan requires an explicit communicator when num_ranks()>1; "
        "use global_transform_inclusive_scan(..., comm) for collective semantics.");
}

/// @brief Transform inclusive scan with communicator for cross-rank prefix
/// @details Applies transform, computes local inclusive scan, then uses MPI_Exscan
///          to get the prefix from prior ranks and offsets all local results.
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename UnaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<void> transform_inclusive_scan(ExecutionPolicy&& policy,
                                      const Container& input,
                                      OutputContainer& output,
                                      T init,
                                      BinaryOp binary_op,
                                      UnaryOp unary_op,
                                      Comm& comm) {
    (void)policy;

    // Verify the operation is supported for distributed scan.
    static_assert(std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>,
                  "transform_inclusive_scan: Only std::plus<> is supported for distributed scan. "
                  "For non-plus operations, use local scan or implement a custom cross-rank prefix exchange.");

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Step 1: Compute local transform-inclusive-scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        running = binary_op(running, unary_op(local_in[i]));
        local_out[i] = running;
    }

    // Step 2: For single-rank case, we're done
    if (comm.size() <= 1) {
        return {};
    }

    // Step 3: Get exclusive prefix of local sums from prior ranks
    T local_sum = running;
    T prefix = comm.template exscan_sum_value<T>(local_sum);
    // exscan_sum_value returns T{} for rank 0

    // Step 4: Offset all local results by the prefix from prior ranks
    if (comm.rank() > 0) {
        for (size_type i = 0; i < local_out.size(); ++i) {
            local_out[i] = binary_op(prefix, local_out[i]);
        }
    }

    return {};
}

// ============================================================================
// Transform Exclusive Scan
// ============================================================================

/// @brief Transform then compute exclusive scan
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> transform_exclusive_scan(ExecutionPolicy&& policy,
                                      const Container& input,
                                      OutputContainer& output,
                                      T init,
                                      BinaryOp binary_op,
                                      UnaryOp unary_op) {
    (void)policy;

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Compute local transform-exclusive-scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_out[i] = running;
        running = binary_op(running, unary_op(local_in[i]));
    }

    // Step 2: For single-rank case, we're done
    if (input.num_ranks() <= 1) {
        return {};
    }

    // Step 3: Multi-rank requires explicit communicator.
    return make_error<void>(
        status_code::precondition_failed,
        "transform_exclusive_scan requires an explicit communicator when num_ranks()>1; "
        "use global_transform_exclusive_scan(..., comm) for collective semantics.");
}

/// @brief Transform exclusive scan with communicator for cross-rank prefix
/// @details Applies transform, computes local exclusive scan, then uses MPI_Exscan
///          to get the prefix from prior ranks and offsets all local results.
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename UnaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<void> transform_exclusive_scan(ExecutionPolicy&& policy,
                                      const Container& input,
                                      OutputContainer& output,
                                      T init,
                                      BinaryOp binary_op,
                                      UnaryOp unary_op,
                                      Comm& comm) {
    (void)policy;

    // Verify the operation is supported for distributed scan.
    static_assert(std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>,
                  "transform_exclusive_scan: Only std::plus<> is supported for distributed scan. "
                  "For non-plus operations, use local scan or implement a custom cross-rank prefix exchange.");

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // Step 1: Compute local transform-exclusive-scan
    T running = init;
    for (size_type i = 0; i < local_in.size(); ++i) {
        local_out[i] = running;
        running = binary_op(running, unary_op(local_in[i]));
    }

    // Step 2: For single-rank case, we're done
    if (comm.size() <= 1) {
        return {};
    }

    // Step 3: Get exclusive prefix of local sums from prior ranks
    // running now contains init op all_local_transformed_elements
    T local_sum = running;
    T prefix = comm.template exscan_sum_value<T>(local_sum);
    // exscan_sum_value returns T{} for rank 0

    // Step 4: Offset all local results by the prefix from prior ranks
    if (comm.rank() > 0) {
        for (size_type i = 0; i < local_out.size(); ++i) {
            local_out[i] = binary_op(prefix, local_out[i]);
        }
    }

    return {};
}

    // ============================================================================
    // Explicit global_* root APIs (Phase 02 semantic split)
    // ============================================================================

    template <typename ExecutionPolicy, typename Container, typename OutputContainer,
            typename T, typename BinaryOp, typename Comm>
        requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
    result<void> global_inclusive_scan(ExecutionPolicy&& policy,
                             const Container& input,
                             OutputContainer& output,
                             T init,
                             BinaryOp binary_op,
                             Comm& comm) {
        return inclusive_scan(std::forward<ExecutionPolicy>(policy), input, output,
                      init, std::move(binary_op), comm);
    }

    template <typename ExecutionPolicy, typename Container, typename OutputContainer,
            typename T, typename BinaryOp, typename Comm>
        requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
    result<void> global_exclusive_scan(ExecutionPolicy&& policy,
                             const Container& input,
                             OutputContainer& output,
                             T init,
                             BinaryOp binary_op,
                             Comm& comm) {
        return exclusive_scan(std::forward<ExecutionPolicy>(policy), input, output,
                      init, std::move(binary_op), comm);
    }

    template <typename ExecutionPolicy, typename Container, typename OutputContainer,
            typename T, typename BinaryOp, typename UnaryOp, typename Comm>
        requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
    result<void> global_transform_inclusive_scan(ExecutionPolicy&& policy,
                                   const Container& input,
                                   OutputContainer& output,
                                   T init,
                                   BinaryOp binary_op,
                                   UnaryOp unary_op,
                                   Comm& comm) {
        return transform_inclusive_scan(std::forward<ExecutionPolicy>(policy), input, output,
                            init, std::move(binary_op), std::move(unary_op), comm);
    }

    template <typename ExecutionPolicy, typename Container, typename OutputContainer,
            typename T, typename BinaryOp, typename UnaryOp, typename Comm>
        requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
    result<void> global_transform_exclusive_scan(ExecutionPolicy&& policy,
                                   const Container& input,
                                   OutputContainer& output,
                                   T init,
                                   BinaryOp binary_op,
                                   UnaryOp unary_op,
                                   Comm& comm) {
        return transform_exclusive_scan(std::forward<ExecutionPolicy>(policy), input, output,
                            init, std::move(binary_op), std::move(unary_op), comm);
    }

// ============================================================================
// Local-Only Scans (No Communication)
// ============================================================================

/// @brief Local inclusive scan (no MPI communication)
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
void local_inclusive_scan(const Container& input,
                          OutputContainer& output,
                          T init,
                          BinaryOp binary_op) {
    auto local_in = input.local_view();
    auto local_out = output.local_view();

    T running = init;
    for (size_type i = 0; i < local_in.size() && i < local_out.size(); ++i) {
        running = binary_op(running, local_in[i]);
        local_out[i] = running;
    }
}

/// @brief Local exclusive scan (no MPI communication)
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
void local_exclusive_scan(const Container& input,
                          OutputContainer& output,
                          T init,
                          BinaryOp binary_op) {
    auto local_in = input.local_view();
    auto local_out = output.local_view();

    T running = init;
    for (size_type i = 0; i < local_in.size() && i < local_out.size(); ++i) {
        local_out[i] = running;
        running = binary_op(running, local_in[i]);
    }
}

// ============================================================================
// Adjacent Difference (Inverse of Scan)
// ============================================================================

/// @brief Compute adjacent differences
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @param policy Execution policy
/// @param input Input container
/// @param output Output container
/// @return Result indicating success or error
///
/// @par Algorithm:
/// output[0] = input[0]
/// output[i] = input[i] - input[i-1] for i > 0
template <typename ExecutionPolicy, typename Container, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> adjacent_difference(ExecutionPolicy&& policy,
                                 const Container& input,
                                 OutputContainer& output) {
    detail::require_collective_comm_or_single_rank(input, "dtl::adjacent_difference");

    (void)policy;

    auto local_in = input.local_view();
    auto local_out = output.local_view();

    if (local_in.size() != local_out.size()) {
        return make_error<void>(status_code::invalid_argument,
                               "Input and output local sizes must match");
    }

    if (local_in.empty()) {
        return {};
    }

    // First element: need previous rank's last element for global correctness
    // For local-only version, just use the input's first element
    local_out[0] = local_in[0];

    // Remaining elements
    for (size_type i = 1; i < local_in.size(); ++i) {
        local_out[i] = local_in[i] - local_in[i - 1];
    }

    // Stub: For multi-rank case, would need to communicate last element
    // from previous rank to adjust local_out[0]

    return {};
}

/// @brief Adjacent difference with default execution
template <typename Container, typename OutputContainer>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> adjacent_difference(const Container& input,
                                 OutputContainer& output) {
    return adjacent_difference(seq{}, input, output);
}

// ============================================================================
// Partial Sum (Alias for Inclusive Scan with Plus)
// ============================================================================

/// @brief Compute partial sums (alias for inclusive_scan with plus)
template <typename ExecutionPolicy, typename Container, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> partial_sum(ExecutionPolicy&& policy,
                         const Container& input,
                         OutputContainer& output) {
    return inclusive_scan(std::forward<ExecutionPolicy>(policy), input, output);
}

/// @brief Partial sum with default execution
template <typename Container, typename OutputContainer>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
result<void> partial_sum(const Container& input, OutputContainer& output) {
    return partial_sum(seq{}, input, output);
}

// ============================================================================
// Async Scan Variants
// ============================================================================

/// @brief Asynchronously compute inclusive prefix scan
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @return Future indicating completion
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
auto async_inclusive_scan(const Container& input,
                          OutputContainer& output,
                          T init,
                          BinaryOp binary_op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = inclusive_scan(seq{}, input, output, init, std::move(binary_op));
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

/// @brief Asynchronously compute exclusive prefix scan
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @return Future indicating completion
template <typename Container, typename OutputContainer, typename T, typename BinaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
auto async_exclusive_scan(const Container& input,
                          OutputContainer& output,
                          T init,
                          BinaryOp binary_op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = exclusive_scan(seq{}, input, output, init, std::move(binary_op));
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

/// @brief Asynchronously compute transform inclusive scan
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @tparam UnaryOp Unary transform operation type
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @param unary_op Transform operation
/// @return Future indicating completion
template <typename Container, typename OutputContainer, typename T, typename BinaryOp, typename UnaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
auto async_transform_inclusive_scan(const Container& input,
                                    OutputContainer& output,
                                    T init,
                                    BinaryOp binary_op,
                                    UnaryOp unary_op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = transform_inclusive_scan(seq{}, input, output, init, std::move(binary_op), std::move(unary_op));
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

/// @brief Asynchronously compute transform exclusive scan
/// @tparam Container Distributed container type
/// @tparam OutputContainer Output container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @tparam UnaryOp Unary transform operation type
/// @param input Input distributed container
/// @param output Output distributed container
/// @param init Initial value
/// @param binary_op Binary operation (associative)
/// @param unary_op Transform operation
/// @return Future indicating completion
template <typename Container, typename OutputContainer, typename T, typename BinaryOp, typename UnaryOp>
    requires DistributedContainer<Container> &&
             DistributedContainer<OutputContainer>
auto async_transform_exclusive_scan(const Container& input,
                                    OutputContainer& output,
                                    T init,
                                    BinaryOp binary_op,
                                    UnaryOp unary_op)
    -> futures::distributed_future<void> {
    auto promise = std::make_shared<futures::distributed_promise<void>>();
    auto future = promise->get_future();

    try {
        auto result = transform_exclusive_scan(seq{}, input, output, init, std::move(binary_op), std::move(unary_op));
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
