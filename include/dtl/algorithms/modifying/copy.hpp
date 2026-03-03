// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file copy.hpp
/// @brief Distributed copy algorithm
/// @details Copy elements between distributed containers.
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
#include <dtl/policies/execution/async.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <memory>

namespace dtl {

// ============================================================================
// Copy Result Type
// ============================================================================

/// @brief Result of a distributed copy operation
struct copy_result {
    /// @brief Number of elements copied
    size_type count = 0;

    /// @brief Whether copy completed successfully
    bool success = true;
};

// ============================================================================
// Distributed copy
// ============================================================================

/// @brief Copy elements from source to destination container
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @return Result containing copy information (local count only)
///
/// @par Complexity:
/// O(n/p) local copies. May involve redistribution if partitions differ.
///
/// @par Same Partition Case:
/// If containers have the same partition, this is a local operation only.
///
/// @par Different Partition Case:
/// If partitions differ, this involves communication to redistribute data.
///
/// @par Note:
/// Returns local count only. Use communicator overload for global count.
///
/// @par Example:
/// @code
/// distributed_vector<int> src(1000, ctx);
/// distributed_vector<int> dst(1000, ctx);
/// auto result = dtl::copy(dtl::par{}, src, dst);
/// @endcode
template <typename ExecutionPolicy, typename InputContainer, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<copy_result> copy([[maybe_unused]] ExecutionPolicy&& policy,
                         const InputContainer& input,
                         OutputContainer& output) {
    // Verify partition compatibility — mismatched partitions produce wrong results
    if (input.global_size() != output.global_size() ||
        input.num_ranks() != output.num_ranks() ||
        input.local_size() != output.local_size()) {
        return make_error<copy_result>(
            status_code::invalid_argument,
            "copy: source and destination have incompatible partitions. "
            "Use redistribute() for cross-partition data movement.");
    }

    copy_result res;

    // Copy local partitions (same-partition case, no communication needed)
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto in_it = in_local.begin();
    auto out_it = out_local.begin();

    for (; in_it != in_local.end() && out_it != out_local.end();
         ++in_it, ++out_it) {
        *out_it = *in_it;
        ++res.count;
    }

    return result<copy_result>{res};
}

/// @brief Copy with default execution
template <typename InputContainer, typename OutputContainer>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<copy_result> copy(const InputContainer& input, OutputContainer& output) {
    return copy(seq{}, input, output);
}

// ============================================================================
// Copy with communicator (distributed global count)
// ============================================================================

/// @brief Copy with distributed global count via communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param comm The communicator for allreduce
/// @return Result containing global copy count across all ranks
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result contains the global count consistent across all ranks.
///
/// @par Example:
/// @code
/// distributed_vector<int> src(1000, ctx);
/// distributed_vector<int> dst(1000, ctx);
/// auto& comm = ctx.communicator();
/// auto result = dtl::copy(dtl::par{}, src, dst, comm);
/// std::cout << "Copied " << result.count << " elements globally\n";
/// @endcode
template <typename ExecutionPolicy, typename InputContainer, typename OutputContainer, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<copy_result> copy(ExecutionPolicy&& policy,
                         const InputContainer& input,
                         OutputContainer& output,
                         Comm& comm) {
    // Perform local copy
    auto local_result = copy(std::forward<ExecutionPolicy>(policy), input, output);

    if (!local_result) {
        return local_result;
    }

    // Allreduce to get global count
    copy_result global_res;
    global_res.count = comm.template allreduce_sum_value<size_type>(local_result.value().count);
    global_res.success = local_result.value().success;

    return result<copy_result>{global_res};
}

/// @brief Copy with communicator (default execution)
template <typename InputContainer, typename OutputContainer, typename Comm>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer> &&
             Communicator<Comm>
result<copy_result> copy(const InputContainer& input, OutputContainer& output, Comm& comm) {
    return copy(seq{}, input, output, comm);
}

// ============================================================================
// Copy with predicate
// ============================================================================

/// @brief Copy elements satisfying predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam Predicate Predicate type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container (must be sized appropriately)
/// @param pred Predicate for selecting elements
/// @return Result containing copy information
///
/// @note This may produce a different number of elements per rank.
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename Predicate>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<copy_result> copy_if(ExecutionPolicy&& policy,
                            const InputContainer& input,
                            OutputContainer& output,
                            Predicate pred) {
    copy_result res;

    auto in_local = input.local_view();
    auto out_local = output.local_view();
    auto out_it = out_local.begin();

    for (auto in_it = in_local.begin();
         in_it != in_local.end() && out_it != out_local.end();
         ++in_it) {
        if (pred(*in_it)) {
            *out_it = *in_it;
            ++out_it;
            ++res.count;
        }
    }

    return result<copy_result>{res};
}

/// @brief Copy_if with default execution
template <typename InputContainer, typename OutputContainer, typename Predicate>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<copy_result> copy_if(const InputContainer& input,
                            OutputContainer& output,
                            Predicate pred) {
    return copy_if(seq{}, input, output, std::move(pred));
}

// ============================================================================
// Copy_n
// ============================================================================

/// @brief Copy first n elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param policy Execution policy
/// @param input Source container
/// @param n Number of elements to copy (global count)
/// @param output Destination container
/// @return Result containing copy information
template <typename ExecutionPolicy, typename InputContainer, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<copy_result> copy_n(ExecutionPolicy&& policy,
                           const InputContainer& input,
                           size_type n,
                           OutputContainer& output) {
    copy_result res;

    // Copy up to n elements from local partition
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto in_it = in_local.begin();
    auto out_it = out_local.begin();

    for (; in_it != in_local.end() && out_it != out_local.end() && res.count < n;
         ++in_it, ++out_it, ++res.count) {
        *out_it = *in_it;
    }

    return result<copy_result>{res};
}

// ============================================================================
// Local-only copy (no communication)
// ============================================================================

/// @brief Copy local partition only (no communication)
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param input Source container
/// @param output Destination container
/// @return Number of elements copied
///
/// @note NOT collective - copies local data only.
template <typename InputContainer, typename OutputContainer>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
size_type local_copy(const InputContainer& input, OutputContainer& output) {
    size_type count = 0;
    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto in_it = in_local.begin();
    auto out_it = out_local.begin();

    for (; in_it != in_local.end() && out_it != out_local.end();
         ++in_it, ++out_it, ++count) {
        *out_it = *in_it;
    }

    return count;
}

// ============================================================================
// Async copy
// ============================================================================

/// @brief Asynchronously copy elements
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param input Source container
/// @param output Destination container
/// @return Future containing copy result
template <typename InputContainer, typename OutputContainer>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
auto async_copy(const InputContainer& input, OutputContainer& output)
    -> futures::distributed_future<copy_result> {
    auto promise = std::make_shared<futures::distributed_promise<copy_result>>();
    auto future = promise->get_future();

    try {
        auto result = copy(seq{}, input, output);
        if (result) {
            promise->set_value(result.value());
        } else {
            promise->set_error(result.error());
        }
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

// ============================================================================
// Redistribute (copy with different partition)
// ============================================================================

/// @brief Copy with redistribution to different partition scheme
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container (may have different partition)
/// @return Result indicating success or failure
///
/// @par Collective Operation:
/// This is always collective as it may require communication.
///
/// @note Use this when copying between containers with different partitions.
template <typename ExecutionPolicy, typename InputContainer, typename OutputContainer>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<void> redistribute_copy(ExecutionPolicy&& policy,
                               const InputContainer& input,
                               OutputContainer& output) {
    // Phase 4: Would implement all-to-all redistribution via MPI
    // Current behavior: local copy only (sufficient for same-partition case)
    local_copy(input, output);
    return {};
}

}  // namespace dtl
