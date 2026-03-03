// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file inner_product.hpp
/// @brief Distributed inner_product (dot product) algorithm
/// @details Compute the inner product of two distributed containers.
///          Local inner products are computed via std::inner_product,
///          then allreduced across ranks when a communicator is provided.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/concepts.hpp>
#include <dtl/error/result.hpp>
#include <dtl/algorithms/concepts.hpp>
#include <dtl/algorithms/detail/multi_rank_guard.hpp>
#include <dtl/algorithms/dispatch.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <functional>
#include <numeric>

namespace dtl {

// ============================================================================
// Inner Product (local, no communication)
// ============================================================================

/// @brief Compute inner product of two distributed containers (local only)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type for the result
/// @param policy Execution policy
/// @param a First container
/// @param b Second container
/// @param init Initial value for accumulation
/// @return Local inner product: init + sum(a[i] * b[i]) for local elements
///
/// @par Complexity:
/// O(n/p) local multiply-add operations. No communication.
///
/// @note For global inner product across all ranks, use the communicator overload.
///
/// @par Example:
/// @code
/// distributed_vector<double> a(1000, ctx);
/// distributed_vector<double> b(1000, ctx);
/// double local_dot = dtl::inner_product(dtl::seq{}, a, b, 0.0);
/// @endcode
template <typename ExecutionPolicy, typename Container1,
          typename Container2, typename T>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2>
T inner_product([[maybe_unused]] ExecutionPolicy&& policy,
                const Container1& a,
                const Container2& b,
                T init) {
    detail::require_collective_comm_or_single_rank(a, "dtl::inner_product");
    detail::require_collective_comm_or_single_rank(b, "dtl::inner_product");

    auto a_local = a.local_view();
    auto b_local = b.local_view();

    return std::inner_product(a_local.begin(), a_local.end(),
                              b_local.begin(), init);
}

/// @brief Inner product with default execution
template <typename Container1, typename Container2, typename T>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2>
DTL_DEPRECATED_MSG("inner_product(a, b, init) is local-only for multi-rank containers; use inner_product(..., comm) for global semantics")
T inner_product(const Container1& a, const Container2& b, T init) {
    return inner_product(seq{}, a, b, init);
}

// ============================================================================
// Inner Product with Custom Operations
// ============================================================================

/// @brief Compute inner product with custom binary operations
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type for the result
/// @tparam BinaryOp1 Accumulation operation
/// @tparam BinaryOp2 Element combination operation
/// @param policy Execution policy
/// @param a First container
/// @param b Second container
/// @param init Initial value
/// @param op1 Accumulation operation (default: std::plus<>)
/// @param op2 Element combination operation (default: std::multiplies<>)
/// @return Local result: fold(op1, init, op2(a[i], b[i])...)
///
/// @par Example:
/// @code
/// // Manhattan-style "inner product": sum of max(|a[i]|, |b[i]|)
/// auto result = dtl::inner_product(dtl::seq{}, a, b, 0.0,
///     std::plus<>{},
///     [](double x, double y) { return std::max(std::abs(x), std::abs(y)); });
/// @endcode
template <typename ExecutionPolicy, typename Container1,
          typename Container2, typename T,
          typename BinaryOp1, typename BinaryOp2>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2>
T inner_product([[maybe_unused]] ExecutionPolicy&& policy,
                const Container1& a,
                const Container2& b,
                T init,
                BinaryOp1 op1,
                BinaryOp2 op2) {
    detail::require_collective_comm_or_single_rank(a, "dtl::inner_product");
    detail::require_collective_comm_or_single_rank(b, "dtl::inner_product");

    auto a_local = a.local_view();
    auto b_local = b.local_view();

    return std::inner_product(a_local.begin(), a_local.end(),
                              b_local.begin(), init,
                              std::move(op1), std::move(op2));
}

/// @brief Inner product with custom ops and default execution
template <typename Container1, typename Container2, typename T,
          typename BinaryOp1, typename BinaryOp2>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2>
DTL_DEPRECATED_MSG("inner_product(a, b, init, op1, op2) is local-only for multi-rank containers; use inner_product(..., comm) for global semantics")
T inner_product(const Container1& a, const Container2& b, T init,
                BinaryOp1 op1, BinaryOp2 op2) {
    return inner_product(seq{}, a, b, init, std::move(op1), std::move(op2));
}

// ============================================================================
// Inner Product with Communicator (distributed global result)
// ============================================================================

/// @brief Compute global inner product across all ranks
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type for the result
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param a First container
/// @param b Second container
/// @param init Initial value
/// @param comm Communicator for allreduce
/// @return Global inner product (consistent across all ranks)
///
/// @par Three-Phase Pattern:
/// 1. Local inner product via std::inner_product
/// 2. MPI_Allreduce to sum local contributions
/// 3. Add init to the global sum
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename Container1,
          typename Container2, typename T, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2> &&
             Communicator<Comm>
T inner_product(ExecutionPolicy&& policy,
                const Container1& a,
                const Container2& b,
                T init,
                Comm& comm) {
    // Phase 1: Local inner product (with zero init to avoid counting init per rank)
    T local_result = inner_product(std::forward<ExecutionPolicy>(policy),
                                   a, b, T{});

    // Phase 2/3: Allreduce sum and add init
    T global_sum = comm.template allreduce_sum_value<T>(local_result);
    return init + global_sum;
}

/// @brief Global inner product with default execution
template <typename Container1, typename Container2, typename T, typename Comm>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2> &&
             Communicator<Comm>
T inner_product(const Container1& a, const Container2& b, T init, Comm& comm) {
    return inner_product(seq{}, a, b, init, comm);
}

// ============================================================================
// Named Algorithm: dot
// ============================================================================

/// @brief Compute dot product (alias for inner_product with std::plus/std::multiplies)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @param policy Execution policy
/// @param a First vector
/// @param b Second vector
/// @return Local dot product
///
/// @par Example:
/// @code
/// double d = dtl::dot(dtl::par{}, vec_a, vec_b);
/// @endcode
template <typename ExecutionPolicy, typename Container1, typename Container2>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2>
auto dot(ExecutionPolicy&& policy, const Container1& a, const Container2& b) {
    using value_type = typename Container1::value_type;
    return inner_product(std::forward<ExecutionPolicy>(policy), a, b, value_type{});
}

/// @brief Dot product with default execution
template <typename Container1, typename Container2>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2>
DTL_DEPRECATED_MSG("dot(a, b) is local-only for multi-rank containers; use dot(..., comm) for global semantics")
auto dot(const Container1& a, const Container2& b) {
    return dot(seq{}, a, b);
}

/// @brief Dot product with communicator (global result)
template <typename ExecutionPolicy, typename Container1,
          typename Container2, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2> &&
             Communicator<Comm>
auto dot(ExecutionPolicy&& policy, const Container1& a,
         const Container2& b, Comm& comm) {
    using value_type = typename Container1::value_type;
    return inner_product(std::forward<ExecutionPolicy>(policy),
                         a, b, value_type{}, comm);
}

// ============================================================================
// Async Inner Product
// ============================================================================

/// @brief Asynchronously compute inner product
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type
/// @param a First container
/// @param b Second container
/// @param init Initial value
/// @return Future containing inner product result
template <typename Container1, typename Container2, typename T>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2>
auto async_inner_product(const Container1& a, const Container2& b, T init)
    -> futures::distributed_future<T> {
    auto promise = std::make_shared<futures::distributed_promise<T>>();
    auto future = promise->get_future();

    try {
        auto result = inner_product(seq{}, a, b, init);
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

}  // namespace dtl
