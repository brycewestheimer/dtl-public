// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file replace.hpp
/// @brief Distributed replace algorithms
/// @details Replace elements matching criteria in distributed containers.
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

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

namespace dtl {

// ============================================================================
// Replace Result Type
// ============================================================================

/// @brief Result of a distributed replace operation
struct replace_result {
    /// @brief Number of elements replaced
    size_type count = 0;
};

// ============================================================================
// Distributed replace
// ============================================================================

/// @brief Replace all occurrences of old_value with new_value
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to modify
/// @param old_value Value to find and replace
/// @param new_value Value to substitute
/// @return Result containing replacement count (local count only)
///
/// @par Complexity:
/// O(n/p) local comparisons and assignments.
/// No communication required.
///
/// @par Note:
/// Returns local count only. Use communicator overload for global count.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto result = dtl::replace(dtl::par{}, vec, 0, 42);
/// std::cout << "Replaced " << result.count << " elements locally\n";
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<replace_result> replace([[maybe_unused]] ExecutionPolicy&& policy,
                               Container& container,
                               const T& old_value,
                               const T& new_value) {
    if (container.num_ranks() > 1) {
        return result<replace_result>::failure(
            status{status_code::not_implemented, no_rank,
                   "replace without communicator is not supported for multi-rank containers; "
                   "use replace(..., comm) or local_replace(...)"});
    }

    replace_result res;

    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (*it == old_value) {
            *it = new_value;
            ++res.count;
        }
    }

    return result<replace_result>{res};
}

/// @brief Replace with default execution
template <typename Container, typename T>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("replace(container, old_value, new_value) is local-only in multi-rank mode; use replace(..., comm) for collective semantics or local_replace(...) for rank-local semantics")
result<replace_result> replace(Container& container,
                               const T& old_value,
                               const T& new_value) {
    return replace(seq{}, container, old_value, new_value);
}

// ============================================================================
// Replace with communicator (distributed global count)
// ============================================================================

/// @brief Replace with distributed global count via communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container Container to modify
/// @param old_value Value to find and replace
/// @param new_value Value to substitute
/// @param comm The communicator for allreduce
/// @return Result containing global replacement count across all ranks
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result contains the global count consistent across all ranks.
///
/// @par Complexity:
/// O(n/p) local comparisons and assignments, plus one allreduce.
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto& comm = ctx.communicator();
/// auto result = dtl::replace(dtl::par{}, vec, 0, 42, comm);
/// std::cout << "Replaced " << result.count << " elements globally\n";
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<replace_result> replace(ExecutionPolicy&& policy,
                               Container& container,
                               const T& old_value,
                               const T& new_value,
                               Comm& comm) {
    // Perform local replacement
    auto local_result = replace(std::forward<ExecutionPolicy>(policy),
                                container, old_value, new_value);

    if (!local_result) {
        return local_result;
    }

    // Allreduce to get global count
    replace_result global_res;
    global_res.count = comm.template allreduce_sum_value<size_type>(local_result.value().count);

    return result<replace_result>{global_res};
}

/// @brief Replace with communicator (default execution)
template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
result<replace_result> replace(Container& container,
                               const T& old_value,
                               const T& new_value,
                               Comm& comm) {
    return replace(seq{}, container, old_value, new_value, comm);
}

// ============================================================================
// Replace_if
// ============================================================================

/// @brief Replace elements satisfying predicate
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container Container to modify
/// @param pred Predicate identifying elements to replace
/// @param new_value Value to substitute
/// @return Result containing replacement count (local count only)
///
/// @par Note:
/// Returns local count only. Use communicator overload for global count.
///
/// @par Example:
/// @code
/// auto result = dtl::replace_if(dtl::par{}, vec,
///     [](int x) { return x < 0; }, 0);  // Replace negatives with 0
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
result<replace_result> replace_if([[maybe_unused]] ExecutionPolicy&& policy,
                                  Container& container,
                                  Predicate pred,
                                  const T& new_value) {
    if (container.num_ranks() > 1) {
        return result<replace_result>::failure(
            status{status_code::not_implemented, no_rank,
                   "replace_if without communicator is not supported for multi-rank containers; "
                   "use replace_if(..., comm) or local_replace_if(...)"});
    }

    replace_result res;

    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            *it = new_value;
            ++res.count;
        }
    }

    return result<replace_result>{res};
}

/// @brief Replace_if with default execution
template <typename Container, typename Predicate, typename T>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("replace_if(container, pred, value) is local-only in multi-rank mode; use replace_if(..., comm) for collective semantics or local_replace_if(...) for rank-local semantics")
result<replace_result> replace_if(Container& container,
                                  Predicate pred,
                                  const T& new_value) {
    return replace_if(seq{}, container, std::move(pred), new_value);
}

// ============================================================================
// Replace_if with communicator (distributed global count)
// ============================================================================

/// @brief Replace_if with distributed global count via communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam T Value type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container Container to modify
/// @param pred Predicate identifying elements to replace
/// @param new_value Value to substitute
/// @param comm The communicator for allreduce
/// @return Result containing global replacement count across all ranks
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result contains the global count consistent across all ranks.
///
/// @par Example:
/// @code
/// auto& comm = ctx.communicator();
/// auto result = dtl::replace_if(dtl::par{}, vec,
///     [](int x) { return x < 0; }, 0, comm);
/// std::cout << "Replaced " << result.count << " negative values globally\n";
/// @endcode
template <typename ExecutionPolicy, typename Container, typename Predicate, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
result<replace_result> replace_if(ExecutionPolicy&& policy,
                                  Container& container,
                                  Predicate pred,
                                  const T& new_value,
                                  Comm& comm) {
    // Perform local replacement
    auto local_result = replace_if(std::forward<ExecutionPolicy>(policy),
                                   container, std::move(pred), new_value);

    if (!local_result) {
        return local_result;
    }

    // Allreduce to get global count
    replace_result global_res;
    global_res.count = comm.template allreduce_sum_value<size_type>(local_result.value().count);

    return result<replace_result>{global_res};
}

/// @brief Replace_if with communicator (default execution)
template <typename Container, typename Predicate, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
result<replace_result> replace_if(Container& container,
                                  Predicate pred,
                                  const T& new_value,
                                  Comm& comm) {
    return replace_if(seq{}, container, std::move(pred), new_value, comm);
}

// ============================================================================
// Replace_copy (non-modifying variant)
// ============================================================================

/// @brief Copy with replacement to different container
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param old_value Value to replace
/// @param new_value Replacement value
/// @return Result containing replacement count
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<replace_result> replace_copy([[maybe_unused]] ExecutionPolicy&& policy,
                                    const InputContainer& input,
                                    OutputContainer& output,
                                    const T& old_value,
                                    const T& new_value) {
    detail::require_collective_comm_or_single_rank(input, "dtl::replace_copy");
    detail::require_collective_comm_or_single_rank(output, "dtl::replace_copy");

    replace_result res;

    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto in_it = in_local.begin();
    auto out_it = out_local.begin();

    for (; in_it != in_local.end() && out_it != out_local.end();
         ++in_it, ++out_it) {
        if (*in_it == old_value) {
            *out_it = new_value;
            ++res.count;
        } else {
            *out_it = *in_it;
        }
    }

    return result<replace_result>{res};
}

/// @brief Replace_copy with default execution
template <typename InputContainer, typename OutputContainer, typename T>
    requires DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
DTL_DEPRECATED_MSG("replace_copy(input, output, ...) is local-only in multi-rank mode; use rank-local semantics explicitly or add communicator-aware flow")
result<replace_result> replace_copy(const InputContainer& input,
                                    OutputContainer& output,
                                    const T& old_value,
                                    const T& new_value) {
    return replace_copy(seq{}, input, output, old_value, new_value);
}

// ============================================================================
// Replace_copy_if
// ============================================================================

/// @brief Copy with conditional replacement
/// @tparam ExecutionPolicy Execution policy type
/// @tparam InputContainer Source container type
/// @tparam OutputContainer Destination container type
/// @tparam Predicate Predicate type
/// @tparam T Value type
/// @param policy Execution policy
/// @param input Source container
/// @param output Destination container
/// @param pred Predicate identifying elements to replace
/// @param new_value Replacement value
/// @return Result containing replacement count
template <typename ExecutionPolicy, typename InputContainer,
          typename OutputContainer, typename Predicate, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<InputContainer> &&
             DistributedContainer<OutputContainer>
result<replace_result> replace_copy_if([[maybe_unused]] ExecutionPolicy&& policy,
                                       const InputContainer& input,
                                       OutputContainer& output,
                                       Predicate pred,
                                       const T& new_value) {
    detail::require_collective_comm_or_single_rank(input, "dtl::replace_copy_if");
    detail::require_collective_comm_or_single_rank(output, "dtl::replace_copy_if");

    replace_result res;

    auto in_local = input.local_view();
    auto out_local = output.local_view();

    auto in_it = in_local.begin();
    auto out_it = out_local.begin();

    for (; in_it != in_local.end() && out_it != out_local.end();
         ++in_it, ++out_it) {
        if (pred(*in_it)) {
            *out_it = new_value;
            ++res.count;
        } else {
            *out_it = *in_it;
        }
    }

    return result<replace_result>{res};
}

// ============================================================================
// Local-only replace (no communication)
// ============================================================================

/// @brief Replace in local partition only
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container Container to modify
/// @param old_value Value to replace
/// @param new_value Replacement value
/// @return Number of elements replaced locally
///
/// @note NOT collective - modifies local data only.
template <typename Container, typename T>
    requires DistributedContainer<Container>
size_type local_replace(Container& container,
                        const T& old_value,
                        const T& new_value) {
    size_type count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (*it == old_value) {
            *it = new_value;
            ++count;
        }
    }
    return count;
}

/// @brief Replace in local partition with predicate
template <typename Container, typename Predicate, typename T>
    requires DistributedContainer<Container>
size_type local_replace_if(Container& container,
                           Predicate pred,
                           const T& new_value) {
    size_type count = 0;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        if (pred(*it)) {
            *it = new_value;
            ++count;
        }
    }
    return count;
}

// ============================================================================
// Async replace
// ============================================================================

/// @brief Asynchronously replace elements
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param container Container to modify
/// @param old_value Value to replace
/// @param new_value Replacement value
/// @return Future containing replacement count
template <typename Container, typename T>
    requires DistributedContainer<Container>
auto async_replace(Container& container,
                   const T& old_value,
                   const T& new_value)
    -> futures::distributed_future<replace_result> {
    auto promise = std::make_shared<futures::distributed_promise<replace_result>>();
    auto future = promise->get_future();

    try {
        auto result = replace(seq{}, container, old_value, new_value);
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

/// @brief Asynchronously replace elements satisfying predicate
/// @tparam Container Distributed container type
/// @tparam Predicate Predicate type
/// @tparam T Value type
/// @param container Container to modify
/// @param pred Predicate identifying elements to replace
/// @param new_value Replacement value
/// @return Future containing replacement count
template <typename Container, typename Predicate, typename T>
    requires DistributedContainer<Container>
auto async_replace_if(Container& container,
                      Predicate pred,
                      const T& new_value)
    -> futures::distributed_future<replace_result> {
    auto promise = std::make_shared<futures::distributed_promise<replace_result>>();
    auto future = promise->get_future();

    try {
        auto result = replace_if(seq{}, container, std::move(pred), new_value);
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

}  // namespace dtl
