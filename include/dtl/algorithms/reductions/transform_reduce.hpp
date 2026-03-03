// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file transform_reduce.hpp
/// @brief Distributed transform_reduce algorithm
/// @details Transform elements then reduce (fused map-reduce).
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
#include <dtl/futures/distributed_future.hpp>

#include <functional>
#include <type_traits>

namespace dtl {

// ============================================================================
// Unary Transform-Reduce
// ============================================================================

/// @brief Transform each element then reduce
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type
/// @tparam UnaryOp Transform operation type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param reduce_op Binary reduction operation (associative, commutative)
/// @param transform_op Unary transform operation
/// @return Global transformed reduction result
///
/// @par Complexity:
/// O(n/p) local transform + reduce operations, plus O(log p) allreduce.
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
///
/// @par Advantages over separate transform + reduce:
/// - Single pass over data (better cache utilization)
/// - No intermediate storage needed
/// - More efficient for GPU execution
///
/// @par Example:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// // Sum of squares
/// int sum_sq = dtl::transform_reduce(dtl::par{}, vec, 0,
///     std::plus<>{},
///     [](int x) { return x * x; });
/// @endcode
template <typename ExecutionPolicy, typename Container,
          typename T, typename BinaryOp, typename UnaryOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
T transform_reduce([[maybe_unused]] ExecutionPolicy&& policy,
                   const Container& container,
                   T init,
                   BinaryOp reduce_op,
                   UnaryOp transform_op) {
    detail::require_collective_comm_or_single_rank(container, "dtl::transform_reduce");

    // Fused transform-reduce on local partition
    T local_result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        local_result = reduce_op(local_result, transform_op(*it));
    }

    // Standalone mode: return local result only
    // For distributed transform_reduce, use overload with communicator parameter
    return local_result;
}

/// @brief Transform-reduce with default execution
template <typename Container, typename T, typename BinaryOp, typename UnaryOp>
    requires DistributedContainer<Container>
DTL_DEPRECATED_MSG("transform_reduce(container, ...) is local-only for multi-rank containers; use transform_reduce(..., comm) for collective semantics")
T transform_reduce(const Container& container,
                   T init,
                   BinaryOp reduce_op,
                   UnaryOp transform_op) {
    return transform_reduce(seq{}, container, init,
                            std::move(reduce_op), std::move(transform_op));
}

// ============================================================================
// Communicator-Aware Transform-Reduce
// ============================================================================

/// @brief Transform each element then reduce with distributed allreduce
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type
/// @tparam UnaryOp Transform operation type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container The distributed container
/// @param init Initial value for reduction
/// @param reduce_op Binary reduction operation
/// @param transform_op Unary transform operation
/// @param comm The communicator for allreduce
/// @return Global transformed reduction result
///
/// @par Collective Operation:
/// This is a collective operation - all ranks must participate.
template <typename ExecutionPolicy, typename Container,
          typename T, typename BinaryOp, typename UnaryOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             Communicator<Comm>
T transform_reduce([[maybe_unused]] ExecutionPolicy&& policy,
                   const Container& container,
                   T init,
                   BinaryOp reduce_op,
                   UnaryOp transform_op,
                   Comm& comm) {
    // Fused transform-reduce on local partition
    T local_result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        local_result = reduce_op(local_result, transform_op(*it));
    }

    // Distributed allreduce
    if constexpr (std::is_same_v<BinaryOp, std::plus<>> ||
                  std::is_same_v<BinaryOp, std::plus<T>>) {
        // For sum: subtract init to avoid counting it multiple times
        T local_contribution = local_result - init;
        T global_sum = comm.template allreduce_sum_value<T>(local_contribution);
        return init + global_sum;
    } else if constexpr (std::is_same_v<BinaryOp, std::multiplies<>> ||
                         std::is_same_v<BinaryOp, std::multiplies<T>>) {
        // For product: compute local product WITHOUT init, allreduce, then apply init
        T local_contrib = T{1};
        auto lv = container.local_view();
        for (auto it = lv.begin(); it != lv.end(); ++it) {
            local_contrib = local_contrib * transform_op(*it);
        }
        T global_prod = comm.template allreduce_prod_value<T>(local_contrib);
        return init * global_prod;
    } else {
        // No silent fallback — unsupported operations must error at compile time.
        static_assert(std::is_same_v<BinaryOp, std::plus<>>,
                      "transform_reduce: BinaryOp does not match a known distributed reduction. "
                      "Use std::plus<> or std::multiplies<>. For other operations, use "
                      "local_transform_reduce() or implement a custom communicator reduction.");
        return local_result;  // unreachable
    }
}

/// @brief Transform-reduce with communicator (default execution)
template <typename Container, typename T, typename BinaryOp, typename UnaryOp, typename Comm>
    requires DistributedContainer<Container> &&
             Communicator<Comm>
T transform_reduce(const Container& container,
                   T init,
                   BinaryOp reduce_op,
                   UnaryOp transform_op,
                   Comm& comm) {
    return transform_reduce(seq{}, container, init,
                            std::move(reduce_op), std::move(transform_op), comm);
}

// ============================================================================
// Binary Transform-Reduce (Inner Product)
// ============================================================================

/// @brief Transform pairs of elements then reduce (inner product pattern)
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type
/// @tparam BinaryReduceOp Reduction operation type
/// @tparam BinaryTransformOp Transform operation type
/// @param policy Execution policy
/// @param container1 First container
/// @param container2 Second container
/// @param init Initial value
/// @param reduce_op Reduction operation
/// @param transform_op Binary transform operation
/// @return Global transformed reduction result
///
/// @par Inner Product Specialization:
/// With default operations (plus, multiplies), computes dot product.
///
/// @par Requirements:
/// - Containers must have compatible partitioning
/// - Local sizes must match
///
/// @par Example:
/// @code
/// // Dot product
/// double dot = dtl::transform_reduce(dtl::par{}, a, b, 0.0,
///     std::plus<>{}, std::multiplies<>{});
/// @endcode
template <typename ExecutionPolicy, typename Container1, typename Container2,
          typename T, typename BinaryReduceOp, typename BinaryTransformOp>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2>
T transform_reduce([[maybe_unused]] ExecutionPolicy&& policy,
                   const Container1& container1,
                   const Container2& container2,
                   T init,
                   BinaryReduceOp reduce_op,
                   BinaryTransformOp transform_op) {
    detail::require_collective_comm_or_single_rank(container1, "dtl::transform_reduce");
    detail::require_collective_comm_or_single_rank(container2, "dtl::transform_reduce");

    T local_result = init;

    auto local1 = container1.local_view();
    auto local2 = container2.local_view();

    auto it1 = local1.begin();
    auto it2 = local2.begin();

    for (; it1 != local1.end() && it2 != local2.end(); ++it1, ++it2) {
        local_result = reduce_op(local_result, transform_op(*it1, *it2));
    }

    // Standalone mode: return local result only
    return local_result;
}

/// @brief Binary transform-reduce with default execution
template <typename Container1, typename Container2,
          typename T, typename BinaryReduceOp, typename BinaryTransformOp>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2>
DTL_DEPRECATED_MSG("transform_reduce(a, b, ...) is local-only for multi-rank containers; use transform_reduce(..., comm) for collective semantics")
T transform_reduce(const Container1& container1,
                   const Container2& container2,
                   T init,
                   BinaryReduceOp reduce_op,
                   BinaryTransformOp transform_op) {
    return transform_reduce(seq{}, container1, container2, init,
                            std::move(reduce_op), std::move(transform_op));
}

/// @brief Binary transform-reduce (inner product) with distributed allreduce
/// @tparam ExecutionPolicy Execution policy type
/// @tparam Container1 First container type
/// @tparam Container2 Second container type
/// @tparam T Value type
/// @tparam BinaryReduceOp Reduction operation type
/// @tparam BinaryTransformOp Transform operation type
/// @tparam Comm Communicator type
/// @param policy Execution policy
/// @param container1 First container
/// @param container2 Second container
/// @param init Initial value
/// @param reduce_op Reduction operation
/// @param transform_op Binary transform operation
/// @param comm The communicator for allreduce
/// @return Global transformed reduction result
template <typename ExecutionPolicy, typename Container1, typename Container2,
          typename T, typename BinaryReduceOp, typename BinaryTransformOp, typename Comm>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container1> &&
             DistributedContainer<Container2> &&
             Communicator<Comm>
T transform_reduce(ExecutionPolicy&& policy,
                   const Container1& container1,
                   const Container2& container2,
                   T init,
                   BinaryReduceOp reduce_op,
                   BinaryTransformOp transform_op,
                   Comm& comm) {
    T local_result = init;

    auto local1 = container1.local_view();
    auto local2 = container2.local_view();

    auto it1 = local1.begin();
    auto it2 = local2.begin();

    for (; it1 != local1.end() && it2 != local2.end(); ++it1, ++it2) {
        local_result = reduce_op(local_result, transform_op(*it1, *it2));
    }

    // Distributed allreduce
    if constexpr (std::is_same_v<BinaryReduceOp, std::plus<>> ||
                  std::is_same_v<BinaryReduceOp, std::plus<T>>) {
        T local_contribution = local_result - init;
        T global_sum = comm.template allreduce_sum_value<T>(local_contribution);
        return init + global_sum;
    } else if constexpr (std::is_same_v<BinaryReduceOp, std::multiplies<>> ||
                         std::is_same_v<BinaryReduceOp, std::multiplies<T>>) {
        // For product: compute local product WITHOUT init, allreduce, then apply init
        T local_contrib = T{1};
        auto lv1 = container1.local_view();
        auto lv2 = container2.local_view();
        auto i1 = lv1.begin();
        auto i2 = lv2.begin();
        for (; i1 != lv1.end() && i2 != lv2.end(); ++i1, ++i2) {
            local_contrib = local_contrib * transform_op(*i1, *i2);
        }
        T global_prod = comm.template allreduce_prod_value<T>(local_contrib);
        return init * global_prod;
    } else {
        // No silent fallback — unsupported operations must error at compile time.
        static_assert(std::is_same_v<BinaryReduceOp, std::plus<>>,
                      "transform_reduce (binary): BinaryReduceOp does not match a known "
                      "distributed reduction. Use std::plus<> or std::multiplies<>.");
        return local_result;  // unreachable
    }
}

/// @brief Binary transform-reduce with communicator (default execution)
template <typename Container1, typename Container2,
          typename T, typename BinaryReduceOp, typename BinaryTransformOp, typename Comm>
    requires DistributedContainer<Container1> &&
             DistributedContainer<Container2> &&
             Communicator<Comm>
T transform_reduce(const Container1& container1,
                   const Container2& container2,
                   T init,
                   BinaryReduceOp reduce_op,
                   BinaryTransformOp transform_op,
                   Comm& comm) {
    return transform_reduce(seq{}, container1, container2, init,
                            std::move(reduce_op), std::move(transform_op), comm);
}

// Note: inner_product overloads moved to inner_product.hpp (Phase R6)

// ============================================================================
// Local-only transform-reduce (no communication)
// ============================================================================

/// @brief Transform-reduce local partition only (no communication)
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type
/// @tparam UnaryOp Transform operation type
/// @param container The distributed container
/// @param init Initial value
/// @param reduce_op Reduction operation
/// @param transform_op Transform operation
/// @return Local transformed reduction result
template <typename Container, typename T, typename BinaryOp, typename UnaryOp>
    requires DistributedContainer<Container>
T local_transform_reduce(const Container& container,
                         T init,
                         BinaryOp reduce_op,
                         UnaryOp transform_op) {
    T result = init;
    auto local_view = container.local_view();
    for (auto it = local_view.begin(); it != local_view.end(); ++it) {
        result = reduce_op(result, transform_op(*it));
    }
    return result;
}

// ============================================================================
// Async transform-reduce
// ============================================================================

/// @brief Asynchronously transform-reduce (single-rank)
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type
/// @tparam UnaryOp Transform operation type
/// @param container The distributed container
/// @param init Initial value
/// @param reduce_op Reduction operation
/// @param transform_op Transform operation
/// @return Future containing the local transform-reduce result
template <typename Container, typename T, typename BinaryOp, typename UnaryOp>
    requires DistributedContainer<Container>
auto async_transform_reduce(const Container& container,
                            T init,
                            BinaryOp reduce_op,
                            UnaryOp transform_op)
    -> futures::distributed_future<T> {
    detail::require_collective_comm_or_single_rank(container, "dtl::async_transform_reduce");

    auto promise = std::make_shared<futures::distributed_promise<T>>();
    auto future = promise->get_future();

    try {
        T result = transform_reduce(seq{}, container, std::move(init),
                                    std::move(reduce_op), std::move(transform_op));
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

/// @brief Asynchronously transform-reduce with communicator
/// @tparam Container Distributed container type
/// @tparam T Value type
/// @tparam BinaryOp Reduction operation type
/// @tparam UnaryOp Transform operation type
/// @tparam Comm Communicator type
/// @param container The distributed container
/// @param init Initial value
/// @param reduce_op Reduction operation
/// @param transform_op Transform operation
/// @param comm The communicator for allreduce
/// @return Future containing the global transform-reduce result
template <typename Container, typename T, typename BinaryOp,
          typename UnaryOp, typename Comm>
    requires DistributedContainer<Container> && Communicator<Comm>
auto async_transform_reduce(const Container& container,
                            T init,
                            BinaryOp reduce_op,
                            UnaryOp transform_op,
                            Comm& comm)
    -> futures::distributed_future<T> {
    auto promise = std::make_shared<futures::distributed_promise<T>>();
    auto future = promise->get_future();

    try {
        T result = transform_reduce(seq{}, container, std::move(init),
                                    std::move(reduce_op), std::move(transform_op), comm);
        promise->set_value(std::move(result));
    } catch (...) {
        promise->set_error(status(status_code::unknown_error));
    }

    return future;
}

// ============================================================================
// Common transform-reduce patterns
// ============================================================================

/// @brief Sum of squares
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto sum_of_squares(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return transform_reduce(std::forward<ExecutionPolicy>(policy),
                            container, value_type{},
                            std::plus<>{},
                            [](const value_type& x) { return x * x; });
}

/// @brief Sum of absolute values (L1 norm)
template <typename ExecutionPolicy, typename Container>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container>
auto sum_of_abs(ExecutionPolicy&& policy, const Container& container) {
    using value_type = typename Container::value_type;
    return transform_reduce(std::forward<ExecutionPolicy>(policy),
                            container, value_type{},
                            std::plus<>{},
                            [](const value_type& x) { return x < value_type{} ? -x : x; });
}

}  // namespace dtl

// Backward compatibility: inner_product was historically defined in this file.
// Include the dedicated header so existing code that includes transform_reduce.hpp
// continues to find dtl::inner_product.
#include <dtl/algorithms/reductions/inner_product.hpp>
