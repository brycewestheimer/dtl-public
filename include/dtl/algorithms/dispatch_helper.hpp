// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file dispatch_helper.hpp
/// @brief Generic algorithm dispatch template using constexpr-if
/// @details Provides dispatch_algorithm<Policy, SeqImpl, ParImpl, AsyncImpl>
///          that selects the correct implementation based on execution policy
///          type at compile time. This consolidates the repeated dispatch
///          pattern used across algorithm files.
/// @since 0.1.0
/// @see dispatch.hpp for the lower-level execution_dispatcher specializations

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/policies/execution/execution_policy.hpp>
#include <dtl/policies/execution/seq.hpp>
#include <dtl/policies/execution/par.hpp>
#include <dtl/policies/execution/async.hpp>
#include <dtl/algorithms/dispatch.hpp>

#include <type_traits>
#include <utility>

namespace dtl {

// ============================================================================
// Generic Algorithm Dispatch
// ============================================================================

/// @brief Dispatch to one of three implementations based on execution policy
/// @tparam Policy Execution policy type (seq, par, async, or cuda_exec)
/// @tparam SeqImpl Callable invoked for sequential execution
/// @tparam ParImpl Callable invoked for parallel execution
/// @tparam AsyncImpl Callable invoked for async execution
/// @param seq_impl The sequential implementation
/// @param par_impl The parallel implementation
/// @param async_impl The async implementation
/// @return The result of whichever implementation is selected
///
/// @par Example:
/// @code
/// template <typename ExecutionPolicy, typename Container, typename T>
/// result<void> my_algorithm(ExecutionPolicy&&, Container& c, const T& val) {
///     using Policy = std::decay_t<ExecutionPolicy>;
///     return dispatch_algorithm<Policy>(
///         [&]() { /* seq implementation */ return result<void>::success(); },
///         [&]() { /* par implementation */ return result<void>::success(); },
///         [&]() { /* async implementation */ return result<void>::success(); }
///     );
/// }
/// @endcode
template <typename Policy, typename SeqImpl, typename ParImpl, typename AsyncImpl>
auto dispatch_algorithm(SeqImpl&& seq_impl, ParImpl&& par_impl, AsyncImpl&& async_impl)
    -> decltype(std::forward<SeqImpl>(seq_impl)()) {
    if constexpr (is_seq_policy_v<Policy>) {
        return std::forward<SeqImpl>(seq_impl)();
    } else if constexpr (is_par_policy_v<Policy>) {
        return std::forward<ParImpl>(par_impl)();
    } else if constexpr (is_async_policy_v<Policy>) {
        return std::forward<AsyncImpl>(async_impl)();
    } else {
        // Default fallback: treat unknown policies as sequential
        return std::forward<SeqImpl>(seq_impl)();
    }
}

/// @brief Dispatch to seq or par implementation (no async variant)
/// @tparam Policy Execution policy type
/// @tparam SeqImpl Callable for sequential execution
/// @tparam ParImpl Callable for parallel execution
/// @param seq_impl The sequential implementation
/// @param par_impl The parallel implementation
/// @return The result of whichever implementation is selected
///
/// @details For algorithms that do not have an async variant, this two-way
///          dispatch is sufficient. If an async policy is passed, it falls
///          back to sequential execution (matching the existing behavior in
///          execution_dispatcher<async>).
template <typename Policy, typename SeqImpl, typename ParImpl>
auto dispatch_algorithm(SeqImpl&& seq_impl, ParImpl&& par_impl)
    -> decltype(std::forward<SeqImpl>(seq_impl)()) {
    if constexpr (is_par_policy_v<Policy>) {
        return std::forward<ParImpl>(par_impl)();
    } else {
        // seq, async, and unknown all use sequential path
        return std::forward<SeqImpl>(seq_impl)();
    }
}

/// @brief Concept for types that can be used as dispatch implementations
/// @details Requires the callable to be invocable with no arguments.
template <typename F>
concept DispatchImpl = std::invocable<F>;

}  // namespace dtl
