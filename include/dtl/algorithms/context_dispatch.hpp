// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file context_dispatch.hpp
/// @brief Context-aware algorithm overloads for explicit MPI-based collective dispatch
/// @details These overloads extract the MPI communicator from a context and forward
///          to the underlying generic distributed algorithms. NCCL-backed generic
///          algorithm dispatch is intentionally not selected here because the current
///          algorithm layer is not device-buffer-aware.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/context.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/algorithms/reductions/reduce.hpp>
#include <dtl/algorithms/reductions/scan.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

#include <functional>
#include <type_traits>

namespace dtl {

// ============================================================================
// Context Communicator Extraction
// ============================================================================

namespace detail {

/// @brief Extract the generic multi-rank communicator from a context
/// @details Generic distributed algorithms remain MPI-primary even when an NCCL
///          domain is present in the same context.
template <typename Ctx>
auto& get_comm_adapter(Ctx& ctx) {
    if constexpr (Ctx::template has<mpi_domain>()) {
        return ctx.template get<mpi_domain>().communicator();
    } else {
        static_assert(Ctx::template has<mpi_domain>(),
                      "Context must contain an MPI domain for generic collective operations");
    }
}

template <typename Ctx>
const auto& get_comm_adapter(const Ctx& ctx) {
    if constexpr (Ctx::template has<mpi_domain>()) {
        return ctx.template get<mpi_domain>().communicator();
    } else {
        static_assert(Ctx::template has<mpi_domain>(),
                      "Context must contain an MPI domain for generic collective operations");
    }
}

/// @brief Trait to check if a type is a context (not a communicator)
template <typename T>
struct is_context : std::false_type {};

template <typename... Domains>
struct is_context<context<Domains...>> : std::true_type {};

template <typename T>
inline constexpr bool is_context_v = is_context<std::decay_t<T>>::value;

}  // namespace detail

// ============================================================================
// Context-Aware Reduce
// ============================================================================

/// @brief Reduce with context using the context's MPI communicator
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp,
          typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T reduce(ExecutionPolicy&& policy,
         const Container& container,
         T init,
         BinaryOp op,
         Ctx& ctx) {
    auto& comm = detail::get_comm_adapter(ctx);
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, op, comm);
}

/// @brief Reduce with context (default plus operation)
template <typename ExecutionPolicy, typename Container, typename T, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T reduce(ExecutionPolicy&& policy, const Container& container, T init, Ctx& ctx) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{}, ctx);
}

/// @brief Sum with context
template <typename ExecutionPolicy, typename Container, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
auto sum(ExecutionPolicy&& policy, const Container& container, Ctx& ctx) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{}, std::plus<>{}, ctx);
}

/// @brief Product with context
template <typename ExecutionPolicy, typename Container, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
auto product(ExecutionPolicy&& policy, const Container& container, Ctx& ctx) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  value_type{1}, std::multiplies<>{}, ctx);
}

/// @brief Min element with context
template <typename ExecutionPolicy, typename Container, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
auto min_element(ExecutionPolicy&& policy, const Container& container, Ctx& ctx) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_min<value_type>::identity(), reduce_min<value_type>{}, ctx);
}

/// @brief Max element with context
template <typename ExecutionPolicy, typename Container, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
auto max_element(ExecutionPolicy&& policy, const Container& container, Ctx& ctx) {
    using value_type = typename Container::value_type;
    return reduce(std::forward<ExecutionPolicy>(policy), container,
                  reduce_max<value_type>::identity(), reduce_max<value_type>{}, ctx);
}

/// @brief Reduce to root with context
template <typename ExecutionPolicy, typename Container, typename T, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
reduce_result<T> reduce_to(ExecutionPolicy&& policy,
                            const Container& container,
                            T init,
                            Ctx& ctx,
                            rank_t root) {
    auto& comm = detail::get_comm_adapter(ctx);
    return reduce_to(std::forward<ExecutionPolicy>(policy), container, init, comm, root);
}

/// @brief Distributed reduce with context
template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp,
          typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
reduce_result<T> distributed_reduce(ExecutionPolicy&& policy,
                                     const Container& container,
                                     T init,
                                     BinaryOp op,
                                     Ctx& ctx) {
    auto& comm = detail::get_comm_adapter(ctx);
    return distributed_reduce(std::forward<ExecutionPolicy>(policy),
                               container, init, op, comm);
}

// ============================================================================
// Context-Aware Scan
// ============================================================================

/// @brief Inclusive scan with context
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             detail::is_context_v<Ctx>
result<void> inclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op,
                            Ctx& ctx) {
    auto& comm = detail::get_comm_adapter(ctx);
    return inclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, init, binary_op, comm);
}

/// @brief Exclusive scan with context
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             detail::is_context_v<Ctx>
result<void> exclusive_scan(ExecutionPolicy&& policy,
                            const Container& input,
                            OutputContainer& output,
                            T init,
                            BinaryOp binary_op,
                            Ctx& ctx) {
    auto& comm = detail::get_comm_adapter(ctx);
    return exclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, init, binary_op, comm);
}

/// @brief Global inclusive scan with context
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             detail::is_context_v<Ctx>
result<void> global_inclusive_scan(ExecutionPolicy&& policy,
                                   const Container& input,
                                   OutputContainer& output,
                                   T init,
                                   BinaryOp binary_op,
                                   Ctx& ctx) {
    return inclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, init, std::move(binary_op), ctx);
}

/// @brief Global exclusive scan with context
template <typename ExecutionPolicy, typename Container, typename OutputContainer,
          typename T, typename BinaryOp, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             DistributedContainer<OutputContainer> &&
             detail::is_context_v<Ctx>
result<void> global_exclusive_scan(ExecutionPolicy&& policy,
                                   const Container& input,
                                   OutputContainer& output,
                                   T init,
                                   BinaryOp binary_op,
                                   Ctx& ctx) {
    return exclusive_scan(std::forward<ExecutionPolicy>(policy),
                          input, output, init, std::move(binary_op), ctx);
}

// ============================================================================
// Context-Aware Global Reduce Aliases
// ============================================================================

template <typename ExecutionPolicy, typename Container, typename T, typename BinaryOp,
          typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T global_reduce(ExecutionPolicy&& policy,
                const Container& container,
                T init,
                BinaryOp op,
                Ctx& ctx) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, op, ctx);
}

template <typename ExecutionPolicy, typename Container, typename T, typename Ctx>
    requires ExecutionPolicyType<ExecutionPolicy> &&
             DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T global_reduce(ExecutionPolicy&& policy,
                const Container& container,
                T init,
                Ctx& ctx) {
    return reduce(std::forward<ExecutionPolicy>(policy), container, init, std::plus<>{}, ctx);
}

template <typename Container, typename T, typename BinaryOp, typename Ctx>
    requires DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T global_reduce(const Container& container,
                T init,
                BinaryOp op,
                Ctx& ctx) {
    return global_reduce(seq{}, container, init, std::move(op), ctx);
}

template <typename Container, typename T, typename Ctx>
    requires DistributedContainer<Container> &&
             detail::is_context_v<Ctx>
T global_reduce(const Container& container,
                T init,
                Ctx& ctx) {
    return global_reduce(seq{}, container, init, std::plus<>{}, ctx);
}

}  // namespace dtl
