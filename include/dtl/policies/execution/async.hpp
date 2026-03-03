// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file async.hpp
/// @brief Asynchronous execution policy
/// @details Non-blocking execution returning a future.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/policies/execution/execution_policy.hpp>

namespace dtl::futures {

// Forward declarations
template <typename T>
class distributed_future;

}  // namespace dtl::futures

namespace dtl {

// Import futures types
using futures::distributed_future;

/// @brief Asynchronous execution policy
/// @details Operations execute asynchronously and return immediately
///          with a future that can be used to wait for completion
///          and retrieve results.
///
/// @par Characteristics:
/// - Non-blocking (asynchronous) initiation
/// - Returns distributed_future<T> instead of T
/// - Allows overlap of computation and communication
/// - Supports continuation chaining via .then()
///
/// @par Usage Pattern:
/// @code
/// auto future = dtl::distributed_reduce(container, dtl::async{}, std::plus<>{});
/// // ... do other work ...
/// auto result = future.get();  // Block and retrieve result
/// @endcode
///
/// @par Use Cases:
/// - Overlapping communication with computation
/// - Pipelining operations
/// - Non-blocking collective operations
struct async {
    /// @brief Policy category tag
    using policy_category = execution_policy_tag;

    /// @brief Get the execution mode
    [[nodiscard]] static constexpr execution_mode mode() noexcept {
        return execution_mode::asynchronous;
    }

    /// @brief Check if execution is blocking
    [[nodiscard]] static constexpr bool is_blocking() noexcept {
        return false;
    }

    /// @brief Check if execution is parallel
    [[nodiscard]] static constexpr bool is_parallel() noexcept {
        return true;  // Typically runs in parallel
    }

    /// @brief Get the parallelism level
    [[nodiscard]] static constexpr parallelism_level parallelism() noexcept {
        return parallelism_level::parallel;
    }

    /// @brief Check if result requires explicit wait
    [[nodiscard]] static constexpr bool requires_wait() noexcept {
        return true;
    }

    /// @brief Check if continuations are supported
    [[nodiscard]] static constexpr bool supports_continuations() noexcept {
        return true;
    }
};

/// @brief Specialization of execution_traits for async
template <>
struct execution_traits<async> {
    static constexpr bool is_blocking = false;
    static constexpr bool is_parallel = true;
    static constexpr execution_mode mode = execution_mode::asynchronous;
    static constexpr parallelism_level parallelism = parallelism_level::parallel;
};

/// @brief Specialization of return type for async policy
template <typename T, typename Future>
struct execution_return_type<async, T, Future> {
    using type = Future;  // Return future instead of T
};

}  // namespace dtl
