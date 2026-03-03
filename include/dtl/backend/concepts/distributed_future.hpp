// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file distributed_future.hpp
/// @brief DistributedFuture concept for async distributed results
/// @details Defines requirements for futures representing distributed computations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <concepts>
#include <functional>
#include <chrono>

namespace dtl {

// ============================================================================
// Future Status
// ============================================================================

/// @brief Status of a distributed future
enum class future_status {
    pending,    ///< Computation not yet complete
    ready,      ///< Result is available
    error,      ///< Computation failed with error
    timeout     ///< Wait timed out
};

// ============================================================================
// Distributed Future Concept
// ============================================================================

/// @brief Core distributed future concept
/// @details Represents the result of an asynchronous distributed computation.
///
/// @par Required Operations:
/// - get(): Block and retrieve result
/// - wait(): Block until ready
/// - valid(): Check if future has a result to retrieve
template <typename T>
concept DistributedFuture = requires(T& fut, const T& cfut) {
    // Value type alias
    typename T::value_type;

    // Block and get result (may throw or return result<>)
    { fut.get() };

    // Block until ready
    { fut.wait() } -> std::same_as<void>;

    // Check if valid (has associated state)
    { cfut.valid() } -> std::same_as<bool>;
};

// ============================================================================
// Pollable Future Concept
// ============================================================================

/// @brief Future that can be polled without blocking
template <typename T>
concept PollableFuture = DistributedFuture<T> &&
    requires(const T& fut) {
    // Non-blocking status check
    { fut.is_ready() } -> std::same_as<bool>;
};

// ============================================================================
// Timed Future Concept
// ============================================================================

/// @brief Future with timeout support
template <typename T>
concept TimedFuture = DistributedFuture<T> &&
    requires(T& fut, std::chrono::milliseconds timeout) {
    // Wait with timeout
    { fut.wait_for(timeout) } -> std::same_as<future_status>;
};

// ============================================================================
// Continuable Future Concept
// ============================================================================

/// @brief Future supporting continuation chaining
/// @details Supports .then() for composing async operations.
template <typename T>
concept ContinuableFuture = DistributedFuture<T> &&
    requires(T& fut) {
    // Then continuation (actual signature depends on callable)
    requires requires(std::function<void(typename T::value_type)> f) {
        { fut.then(f) };
    };
};

// ============================================================================
// Distributed Future Traits
// ============================================================================

/// @brief Traits for distributed future types
template <typename F>
struct distributed_future_traits {
    /// @brief Whether future supports polling
    static constexpr bool supports_polling = false;

    /// @brief Whether future supports timeout
    static constexpr bool supports_timeout = false;

    /// @brief Whether future supports continuations
    static constexpr bool supports_continuations = false;

    /// @brief Whether future participates in collective operations
    static constexpr bool is_collective = false;
};

// ============================================================================
// Future Combinators
// ============================================================================

/// @brief Result of waiting for all futures
template <typename... Ts>
struct when_all_result {
    /// @brief Tuple of results
    std::tuple<Ts...> values;
};

/// @brief Result of waiting for any future
template <typename T>
struct when_any_result {
    /// @brief Index of the completed future
    size_type index{0};

    /// @brief The completed value
    T value{};
};

/// @brief Specialization for void futures
template <>
struct when_any_result<void> {
    /// @brief Index of the completed future
    size_type index{0};
};

// ============================================================================
// Future Factory Functions
// ============================================================================

/// @brief Create a ready future with a value
/// @tparam T Value type
/// @param value The value
/// @return A future that is immediately ready
template <typename T>
struct ready_future {
    using value_type = T;

    /// @brief The ready value
    T value_;

    /// @brief Construct with value
    explicit ready_future(T value) : value_(std::move(value)) {}

    /// @brief Get the value (returns immediately)
    [[nodiscard]] T get() { return std::move(value_); }

    /// @brief Wait (no-op, already ready)
    void wait() const noexcept {}

    /// @brief Check validity
    [[nodiscard]] bool valid() const noexcept { return true; }

    /// @brief Check if ready (always true)
    [[nodiscard]] bool is_ready() const noexcept { return true; }

    /// @brief Wait with timeout (always returns ready immediately)
    [[nodiscard]] future_status wait_for(std::chrono::milliseconds /*timeout*/) const noexcept {
        return future_status::ready;
    }

    /// @brief Add continuation
    template <typename F>
    auto then(F&& func) {
        using ResultType = std::invoke_result_t<F, T>;
        return ready_future<ResultType>(std::forward<F>(func)(value_));
    }
};

/// @brief Traits specialization for ready_future
template <typename T>
struct distributed_future_traits<ready_future<T>> {
    static constexpr bool supports_polling = true;
    static constexpr bool supports_timeout = true;
    static constexpr bool supports_continuations = true;
    static constexpr bool is_collective = false;
};

/// @brief Create a future that represents failure
/// @tparam T Expected value type
template <typename T>
struct failed_future {
    using value_type = T;

    /// @brief The error
    status error_;

    /// @brief Construct with error
    explicit failed_future(status err) : error_(std::move(err)) {}

    /// @brief Get throws the error
    [[nodiscard]] T get() {
        // Stub: would throw or return result<T>
        throw std::runtime_error(error_.message());
    }

    /// @brief Wait (no-op, already complete with error)
    void wait() const noexcept {}

    /// @brief Check validity
    [[nodiscard]] bool valid() const noexcept { return true; }

    /// @brief Check if ready (always true, but with error)
    [[nodiscard]] bool is_ready() const noexcept { return true; }

    /// @brief Wait with timeout
    [[nodiscard]] future_status wait_for(std::chrono::milliseconds /*timeout*/) const noexcept {
        return future_status::error;
    }
};

// ============================================================================
// Helper Factory Functions
// ============================================================================

/// @brief Create a ready future
/// @tparam T Value type
/// @param value The value
/// @return Ready future
template <typename T>
[[nodiscard]] ready_future<std::decay_t<T>> make_ready_future(T&& value) {
    return ready_future<std::decay_t<T>>(std::forward<T>(value));
}

/// @brief Create a failed future
/// @tparam T Expected value type
/// @param error The error status
/// @return Failed future
template <typename T>
[[nodiscard]] failed_future<T> make_failed_future(status error) {
    return failed_future<T>(std::move(error));
}

}  // namespace dtl
