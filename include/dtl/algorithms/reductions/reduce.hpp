// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file reduce.hpp
/// @brief Distributed reduce algorithm
/// @details Combine elements using binary operation across distributed container.
///          This is the flagship distributed algorithm, demonstrating the
///          three-phase pattern and proper segmented iteration.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/algorithms/detail/determinism_guard.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

// Forward declare Communicator concept for disambiguation
#include <dtl/backend/concepts/communicator.hpp>

// Futures for async variants
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

namespace dtl {

// ============================================================================
// Reduce Result Type
// ============================================================================

/// @brief Result type for distributed reduce operations
/// @tparam T The value type being reduced
/// @details Contains both local and global results, allowing callers to
///          distinguish between local-only and distributed reductions.
template <typename T>
struct reduce_result {
    /// @brief This rank's partial result (local reduction)
    T local_value;

    /// @brief Combined result across all ranks (only valid if has_global)
    T global_value;

    /// @brief True if global_value is valid (allreduce was performed)
    bool has_global;

    /// @brief Get the appropriate result value
    /// @return global_value if available, otherwise local_value
    [[nodiscard]] constexpr T value() const noexcept {
        return has_global ? global_value : local_value;
    }

    /// @brief Implicit conversion to value type
    [[nodiscard]] constexpr operator T() const noexcept {
        return value();
    }
};

// ============================================================================
// Communicator-Aware Distributed Reduce (Phase 5 Integration)
// ============================================================================

/// @brief Reduce all elements using MPI allreduce with communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type (std::plus<> for allreduce_sum)
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param op Binary operation (currently only std::plus<> supported for MPI)
/// @param comm The MPI communicator adapter
/// @return Global reduction result (same on all ranks)
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate with the same
/// communicator. The result is identical on all ranks after the allreduce.
///
/// @par Three-Phase Pattern:
/// 1. Local reduction using segmented iteration
/// 2. Barrier synchronization (implicit in allreduce)
/// 3. MPI_Allreduce collective operation
///
/// @par Example:
/// @code
/// mpi::mpi_comm_adapter comm;
/// distributed_vector<int> vec(1000, comm);
/// int sum = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{}, comm);
/// @endcode
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T reduce(ExecutionPolicy&& policy,
         const Container& container,
         T init,
         BinaryOp op,
         Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::reduce");
        !deterministic_guard) {
        throw std::runtime_error(deterministic_guard.error().message());
    }

    // Handle empty container: just return init (no allreduce needed)
    if (container.global_size() == 0) {
        return init;
    }

    // Phase 1: Local reduction using segmented iteration
    // For multiplicative reductions, use identity T{1} to avoid init contamination
    // and eliminate the need for a second pass over the data.
    constexpr bool is_multiply = std::is_same_v<BinaryOp, std::multiplies<>> ||
                                 std::is_same_v<BinaryOp, std::multiplies<T>>;

    const bool force_seq_local = detail::deterministic_mode_enabled() ||
                                 detail::deterministic_policy_requests_fixed_reduction_schedule();
    T local_result = is_multiply ? T{1} : init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            if (force_seq_local) {
                local_result = dispatch_reduce(
                    seq{},
                    segment.begin(), segment.end(),
                    local_result, op);
            } else {
                local_result = dispatch_reduce(
                    std::forward<ExecutionPolicy>(policy),
                    segment.begin(), segment.end(),
                    local_result, op);
            }
        }
    }

    // For additive reductions, subtract init since it was already counted in
    // local reduction and we don't want to add it multiple times across ranks.
    // local_result now contains: init + sum(local_elements)
    // We want global result = init + sum(all_elements)
    // So we allreduce (local_result - init) then add init back.
    T local_contribution = local_result;
    if constexpr (std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>) {
        local_contribution = local_result - init;
    }

    // Phase 2/3: Collective allreduce via MPI
    if constexpr (std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>) {
        T global_sum = comm.template allreduce_sum_value<T>(local_contribution);
        return init + global_sum;
    } else if constexpr (is_multiply) {
        // local_result already contains the product of local elements (with identity T{1}).
        // Single pass — no re-iteration needed.
        T global_prod = comm.template allreduce_prod_value<T>(local_result);
        return init * global_prod;
    } else if constexpr (std::is_same_v<BinaryOp, reduce_min<>> ||
                         std::is_same_v<BinaryOp, reduce_min<T>>) {
        // For min: allreduce the local mins directly
        return comm.template allreduce_min_value<T>(local_result);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_max<>> ||
                         std::is_same_v<BinaryOp, reduce_max<T>>) {
        // For max: allreduce the local maxs directly
        return comm.template allreduce_max_value<T>(local_result);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_land<>> ||
                         std::is_same_v<BinaryOp, reduce_land<T>>) {
        // For logical AND: allreduce the local results
        bool local_bool = static_cast<bool>(local_result);
        bool global_bool = comm.allreduce_land_value(local_bool);
        return static_cast<T>(global_bool);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_lor<>> ||
                         std::is_same_v<BinaryOp, reduce_lor<T>>) {
        // For logical OR: allreduce the local results
        bool local_bool = static_cast<bool>(local_result);
        bool global_bool = comm.allreduce_lor_value(local_bool);
        return static_cast<T>(global_bool);
    } else {
        // No silent fallback — unsupported operations must error at compile time.
        // If you reach this branch, your BinaryOp is not a recognized distributed reduction.
        // Use dtl::reduce_sum, dtl::reduce_min, dtl::reduce_max, std::plus<>, std::multiplies<>,
        // dtl::reduce_land, dtl::reduce_lor, or implement a custom communicator reduction.
        static_assert(std::is_same_v<BinaryOp, std::plus<>>,
                      "reduce: BinaryOp does not match a known distributed reduction. "
                      "Use std::plus<>, std::multiplies<>, dtl::reduce_min, dtl::reduce_max, "
                      "dtl::reduce_land, or dtl::reduce_lor. For other operations, use local_reduce() "
                      "or implement a custom communicator reduction.");
        return local_result;  // unreachable, silences compiler warning
    }
}

/// @brief Reduce with communicator using default plus operation
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T reduce(ExecutionPolicy&& policy, const Container& container, T init, Comm& comm) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{}, comm);
}

/// @brief Reduce to root rank only (MPI_Reduce, not MPI_Allreduce)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param comm The MPI communicator adapter
/// @param root Root rank that receives the result
/// @return reduce_result with global value valid only on root
template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
reduce_result<T> reduce_to(ExecutionPolicy&& policy,
                           const Container& container,
                           T init,
                           Comm& comm,
                           rank_t root) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::reduce_to");
        !deterministic_guard) {
        throw std::runtime_error(deterministic_guard.error().message());
    }

    bool is_root = (comm.rank() == root);

    // Handle empty container
    if (container.global_size() == 0) {
        return reduce_result<T>{init, init, is_root};
    }

    // Phase 1: Local reduction
    const bool force_seq_local = detail::deterministic_mode_enabled() ||
                                 detail::deterministic_policy_requests_fixed_reduction_schedule();
    T local_result = init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            if (force_seq_local) {
                local_result = dispatch_reduce(
                    seq{},
                    segment.begin(), segment.end(),
                    local_result, std::plus<>{});
            } else {
                local_result = dispatch_reduce(
                    std::forward<ExecutionPolicy>(policy),
                    segment.begin(), segment.end(),
                    local_result, std::plus<>{});
            }
        }
    }

    // Phase 2/3: Reduce to root via MPI
    // Subtract init to avoid counting it multiple times
    T local_contribution = local_result - init;
    T global_sum = comm.template reduce_sum_to_root<T>(local_contribution, root);
    T global_result = is_root ? (init + global_sum) : T{};
    return reduce_result<T>{local_result, global_result, is_root};
}

/// @brief Distributed reduce returning reduce_result with communicator
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @tparam Comm Communicator type (must satisfy Communicator concept)
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param op Binary operation
/// @param comm The MPI communicator adapter
/// @return reduce_result containing both local and global values
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
reduce_result<T> distributed_reduce(ExecutionPolicy&& policy,
                                     const Container& container,
                                     T init,
                                     BinaryOp op,
                                     Comm& comm) {
    if (auto deterministic_guard =
            detail::require_deterministic_collective_support(comm.size(), "dtl::distributed_reduce");
        !deterministic_guard) {
        throw std::runtime_error(deterministic_guard.error().message());
    }

    // Handle empty container
    if (container.global_size() == 0) {
        return reduce_result<T>{init, init, true};
    }

    // Phase 1: Local reduction using segmented iteration
    // For multiplicative reductions, use identity T{1} to avoid init contamination
    constexpr bool is_multiply = std::is_same_v<BinaryOp, std::multiplies<>> ||
                                 std::is_same_v<BinaryOp, std::multiplies<T>>;
    const bool force_seq_local = detail::deterministic_mode_enabled() ||
                                 detail::deterministic_policy_requests_fixed_reduction_schedule();
    T local_result = is_multiply ? T{1} : init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            if (force_seq_local) {
                local_result = dispatch_reduce(
                    seq{},
                    segment.begin(), segment.end(),
                    local_result, op);
            } else {
                local_result = dispatch_reduce(
                    std::forward<ExecutionPolicy>(policy),
                    segment.begin(), segment.end(),
                    local_result, op);
            }
        }
    }

    // Phase 2/3: Collective allreduce
    T global_result = local_result;
    if constexpr (std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>) {
        // Subtract init to avoid counting it multiple times
        T local_contribution = local_result - init;
        T global_sum = comm.template allreduce_sum_value<T>(local_contribution);
        global_result = init + global_sum;
    } else if constexpr (is_multiply) {
        // local_result already contains the product of local elements (with identity T{1}).
        // Single pass -- no re-iteration needed.
        T global_prod = comm.template allreduce_prod_value<T>(local_result);
        global_result = init * global_prod;
    } else if constexpr (std::is_same_v<BinaryOp, reduce_min<>> ||
                         std::is_same_v<BinaryOp, reduce_min<T>>) {
        global_result = comm.template allreduce_min_value<T>(local_result);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_max<>> ||
                         std::is_same_v<BinaryOp, reduce_max<T>>) {
        global_result = comm.template allreduce_max_value<T>(local_result);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_land<>> ||
                         std::is_same_v<BinaryOp, reduce_land<T>>) {
        bool local_bool = static_cast<bool>(local_result);
        bool global_bool = comm.allreduce_land_value(local_bool);
        global_result = static_cast<T>(global_bool);
    } else if constexpr (std::is_same_v<BinaryOp, reduce_lor<>> ||
                         std::is_same_v<BinaryOp, reduce_lor<T>>) {
        bool local_bool = static_cast<bool>(local_result);
        bool global_bool = comm.allreduce_lor_value(local_bool);
        global_result = static_cast<T>(global_bool);
    }

    // For multiply, adjust local_result to include init for the reduce_result
    T adjusted_local = is_multiply ? (init * local_result) : local_result;
    return reduce_result<T>{adjusted_local, global_result, true};
}

// ============================================================================
// Distributed reduce (standalone - no communicator)
// ============================================================================

/// @brief Reduce all elements using binary operation
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param op Binary operation (must be associative and commutative)
/// @return Global reduction result
///
/// @par Complexity:
/// O(n/p) local operations, plus O(log p) allreduce communication.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
/// The result is consistent across all ranks.
///
/// @par Three-Phase Pattern:
/// 1. Local reduction using segmented iteration
/// 2. Synchronization at phase boundary
/// 3. Collective allreduce operation
///
/// @par Requirements:
/// - BinaryOp must be associative: op(op(a,b),c) == op(a,op(b,c))
/// - BinaryOp must be commutative: op(a,b) == op(b,a)
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// int sum = dtl::reduce(dtl::par{}, vec, 0, std::plus<>{});
/// int product = dtl::reduce(dtl::par{}, vec, 1, std::multiplies<>{});
/// @endcode
///
/// @note This overload is for standalone (single-rank) usage. For multi-rank
///       usage, pass a communicator as the last argument.
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T reduce(ExecutionPolicy&& policy,
         const Container& container,
         T init,
         BinaryOp op) {
    detail::require_collective_comm_or_single_rank(container, "dtl::reduce");

    // Phase 1: Local reduction using segmented iteration
    T local_result = init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            // Use dispatch for proper execution policy handling
            local_result = dispatch_reduce(
                std::forward<ExecutionPolicy>(policy),
                segment.begin(), segment.end(),
                local_result, op);
        }
    }

    // Phase 2/3: No communicator - return local result only
    // For distributed reduction with MPI, use the overload with communicator parameter
    return local_result;
}

/// @brief Reduce with default addition operation
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @return Sum of all elements plus init
template <typename ExecutionPolicy, typename Container, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T reduce(ExecutionPolicy&& policy, const Container& container, T init) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{});
}

/// @brief Reduce with default execution and addition
template <typename Container, typename T>
    requires DistributedContainer<Container>
T reduce(const Container& container, T init) {
    return reduce(seq{}, container, init);
}

/// @brief Reduce with value-initialized identity
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Sum of all elements
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto reduce(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container, value_type{});
}

// ============================================================================
// Distributed reduce with result type
// ============================================================================

/// @brief Distributed reduce returning reduce_result with local and global values
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param op Binary operation
/// @return reduce_result containing both local and global values
///
/// @par Conformance:
/// - spec.algorithms.distributed_reduce.uses_segmented_local_path
/// - spec.algorithms.distributed_reduce.result_matches_reference
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
reduce_result<T> distributed_reduce(ExecutionPolicy&& policy,
                                     const Container& container,
                                     T init,
                                     BinaryOp op) {
    detail::require_collective_comm_or_single_rank(container, "dtl::distributed_reduce");

    // Phase 1: Local reduction using segmented iteration (CONFORMANCE: uses_segmented_local_path)
    T local_result = init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            local_result = dispatch_reduce(
                std::forward<ExecutionPolicy>(policy),
                segment.begin(), segment.end(),
                local_result, op);
        }
    }

    // Phase 2/3: Collective allreduce (stub - single rank returns same value)
    // Real implementation: global_result = communicator.allreduce(local_result, op);
    T global_result = local_result;

    return reduce_result<T>{local_result, global_result, container.num_ranks() <= 1};
}

// ============================================================================
// Reduce to root (single rank receives result)
// ============================================================================

/// @brief Reduce all elements to a single root rank
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation
/// @param root Rank that receives the result
/// @return reduce_result with has_global true only on root
///
/// @note Only root rank has the valid global result.
/// @note This overload is for standalone usage. For MPI, use the overload with Communicator.
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             (!Communicator<BinaryOp>)  // Disambiguate from MPI version
reduce_result<T> reduce_to(ExecutionPolicy&& policy,
                            const Container& container,
                            T init,
                            BinaryOp op,
                            rank_t root) {
    detail::require_collective_comm_or_single_rank(container, "dtl::reduce_to");

    // Phase 1: Local reduction
    T local_result = init;
    for (auto segment : container.segmented_view()) {
        if (segment.is_local()) {
            local_result = dispatch_reduce(
                std::forward<ExecutionPolicy>(policy),
                segment.begin(), segment.end(),
                local_result, op);
        }
    }

    // Phase 2/3: Reduce to root (stub - single rank always is root)
    // Real implementation: if (my_rank == root) global = reduce_to_root(local, op, root);
    bool is_root = true;  // Stub: single-rank always is root
    return reduce_result<T>{local_result, local_result, is_root};
}

// ============================================================================
// Local-only reduce (no communication)
// ============================================================================

/// @brief Reduce local partition only (no communication)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation
/// @return Local reduction result
///
/// @note NOT collective - reduces local data only. No inter-rank communication.
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T local_reduce(ExecutionPolicy&& policy, const Container& container, T init, BinaryOp op) {
    auto local_v = container.local_view();
    return dispatch_reduce(std::forward<ExecutionPolicy>(policy),
                           local_v.begin(), local_v.end(), init, op);
}

/// @brief Local reduce with default sequential execution
template <typename Container, typename T, typename BinaryOp>
    requires DistributedContainer<Container>
T local_reduce(const Container& container, T init, BinaryOp op) {
    return local_reduce(seq{}, container, init, std::move(op));
}

/// @brief Local reduce with default addition
template <typename Container, typename T>
    requires DistributedContainer<Container>
T local_reduce(const Container& container, T init) {
    return local_reduce(container, init, std::plus<>{});
}

// ============================================================================
// Async reduce
// ============================================================================

/// @brief Asynchronously reduce all elements
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Binary operation type
/// @param container The distributed container
/// @param init Initial value
/// @param op Binary operation
/// @return Future containing global reduction result
template <typename Container, typename T, typename BinaryOp>
    requires DistributedContainer<Container>
auto async_reduce(const Container& container, T init, BinaryOp op)
    -> futures::distributed_future<T> {
    auto promise = std::make_shared<futures::distributed_promise<T>>();
    auto future = promise->get_future();

    try {
        auto result = reduce(seq{}, container, init, std::move(op));
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

// ============================================================================
// Common reduction shortcuts
// ============================================================================

/// @brief Sum all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Global sum
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto sum(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{}, std::plus<>{});
}

/// @brief Sum all elements (default execution)
template <typename Container>
    requires DistributedContainer<Container>
auto sum(const Container& container) {
    return sum(seq{}, container);
}

/// @brief Local sum (no communication)
template <typename Container>
    requires DistributedContainer<Container>
auto local_sum(const Container& container) {
    using value_type = typename Container::value_type;
    return local_reduce(container, value_type{}, std::plus<>{});
}

/// @brief Product of all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Global product
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto product(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{1}, std::multiplies<>{});
}

/// @brief Product of all elements (default execution)
template <typename Container>
    requires DistributedContainer<Container>
auto product(const Container& container) {
    return product(seq{}, container);
}

/// @brief Local product (no communication)
template <typename Container>
    requires DistributedContainer<Container>
auto local_product(const Container& container) {
    using value_type = typename Container::value_type;
    return local_reduce(container, value_type{1}, std::multiplies<>{});
}

/// @brief Minimum of all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Global minimum
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto min_element(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_min<value_type>::identity(), reduce_min<value_type>{});
}

/// @brief Minimum of all elements (default execution)
template <typename Container>
    requires DistributedContainer<Container>
auto min_element(const Container& container) {
    return min_element(seq{}, container);
}

/// @brief Local minimum (no communication)
template <typename Container>
    requires DistributedContainer<Container>
auto local_min(const Container& container) {
    using value_type = typename Container::value_type;
    return local_reduce(container, reduce_min<value_type>::identity(), reduce_min<value_type>{});
}

/// @brief Maximum of all elements
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @param policy Execution policy
/// @param container The distributed container
/// @return Global maximum
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto max_element(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_max<value_type>::identity(), reduce_max<value_type>{});
}

/// @brief Maximum of all elements (default execution)
template <typename Container>
    requires DistributedContainer<Container>
auto max_element(const Container& container) {
    return max_element(seq{}, container);
}

/// @brief Local maximum (no communication)
template <typename Container>
    requires DistributedContainer<Container>
auto local_max(const Container& container) {
    using value_type = typename Container::value_type;
    return local_reduce(container, reduce_max<value_type>::identity(), reduce_max<value_type>{});
}

/// @brief Minimum with communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comm The communicator
/// @return Global minimum across all ranks
template <typename ExecutionPolicy, typename Container, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
auto min_element(ExecutionPolicy&& policy, const Container& container, Comm& comm) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_min<value_type>::identity(), reduce_min<value_type>{}, comm);
}

/// @brief Maximum with communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comm The communicator
/// @return Global maximum across all ranks
template <typename ExecutionPolicy, typename Container, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
auto max_element(ExecutionPolicy&& policy, const Container& container, Comm& comm) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_max<value_type>::identity(), reduce_max<value_type>{}, comm);
}

/// @brief Sum with communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comm The communicator
/// @return Global sum across all ranks
template <typename ExecutionPolicy, typename Container, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
auto sum(ExecutionPolicy&& policy, const Container& container, Comm& comm) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{}, std::plus<>{}, comm);
}

/// @brief Product with communicator (distributed)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param comm The communicator
/// @return Global product across all ranks
template <typename ExecutionPolicy, typename Container, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
auto product(ExecutionPolicy&& policy, const Container& container, Comm& comm) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{1}, std::multiplies<>{}, comm);
}

// ============================================================================
// Explicit global_* root APIs (Phase 02 semantic split)
// ============================================================================

template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T global_reduce(ExecutionPolicy&& policy,
                const Container& container,
                T init,
                BinaryOp op,
                Comm& comm) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, op, comm);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T global_reduce(ExecutionPolicy&& policy,
                const Container& container,
                T init,
                Comm& comm) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{}, comm);
}

template <typename Container, typename T, typename BinaryOp, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
T global_reduce(const Container& container,
                T init,
                BinaryOp op,
                Comm& comm) {
    return global_reduce(seq{}, container, init, std::move(op), comm);
}

template <typename Container, typename T, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
T global_reduce(const Container& container,
                T init,
                Comm& comm) {
    return global_reduce(seq{}, container, init, std::plus<>{}, comm);
}

}  // namespace dtl
