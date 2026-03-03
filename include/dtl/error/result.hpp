// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file result.hpp
/// @brief Result type for operations that may fail
/// @details Provides a monadic result type similar to std::expected<T, status>
///          with methods for safe value extraction and error handling.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/error/status.hpp>
#include <dtl/error/error_type.hpp>

#include <functional>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace dtl {

// =============================================================================
// Result Type
// =============================================================================

/// @brief Result type combining a value or an error status
/// @tparam T The success value type
/// @details Inspired by std::expected<T, E> and Rust's Result<T, E>.
///          Provides a type-safe way to return either a value or an error.
template <typename T>
class result {
public:
    using value_type = T;
    using error_type = status;

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// @brief Default constructor (success with default-constructed value)
    result()
        requires std::default_initializable<T>
        : data_{std::in_place_index<0>, T{}} {}

    /// @brief Construct with success value
    /// @param value The success value
    result(const T& value)
        requires std::copy_constructible<T>
        : data_{std::in_place_index<0>, value} {}

    /// @brief Construct with success value (move)
    /// @param value The success value to move
    result(T&& value)
        requires std::move_constructible<T>
        : data_{std::in_place_index<0>, std::move(value)} {}

    /// @brief Construct with error status
    /// @param error The error status
    result(status error)
        : data_{std::in_place_index<1>, std::move(error)} {
        DTL_ASSERT(error.is_error());
    }

    /// @brief Construct with error status code
    /// @param code The error status code
    result(status_code code)
        : data_{std::in_place_index<1>, status{code}} {
        DTL_ASSERT(code != status_code::ok);
    }

    /// @brief Construct from error object
    /// @param err The error object
    result(const class error& err)
        : data_{std::in_place_index<1>, err.get_status()} {}

    /// @brief Construct from error object (move)
    /// @param err The error object
    result(class error&& err)
        : data_{std::in_place_index<1>, err.get_status()} {}

    /// @brief Construct success value in place
    /// @tparam Args Constructor arguments
    /// @param args Arguments forwarded to T's constructor
    template <typename... Args>
        requires std::constructible_from<T, Args...>
    explicit result(std::in_place_t, Args&&... args)
        : data_{std::in_place_index<0>, std::forward<Args>(args)...} {}

    // -------------------------------------------------------------------------
    // Observers
    // -------------------------------------------------------------------------

    /// @brief Check if result contains a value
    /// @return true if result is a success value
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return data_.index() == 0;
    }

    /// @brief Check if result contains an error
    /// @return true if result is an error
    [[nodiscard]] constexpr bool has_error() const noexcept {
        return data_.index() == 1;
    }

    /// @brief Bool conversion (true if has value)
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return has_value();
    }

    /// @brief Get the success value
    /// @return Reference to the value
    /// @pre has_value() must be true
    [[nodiscard]] T& value() & {
        DTL_ASSERT(has_value());
        return std::get<0>(data_);
    }

    /// @brief Get the success value (const)
    /// @return Const reference to the value
    /// @pre has_value() must be true
    [[nodiscard]] const T& value() const& {
        DTL_ASSERT(has_value());
        return std::get<0>(data_);
    }

    /// @brief Get the success value (rvalue)
    /// @return Rvalue reference to the value
    /// @pre has_value() must be true
    [[nodiscard]] T&& value() && {
        DTL_ASSERT(has_value());
        return std::get<0>(std::move(data_));
    }

    /// @brief Get the error status
    /// @return Reference to the error status
    /// @pre has_error() must be true
    [[nodiscard]] const status& error() const& {
        DTL_ASSERT(has_error());
        return std::get<1>(data_);
    }

    /// @brief Get value or default
    /// @param default_value Value to return if result is error
    /// @return The value if success, otherwise default_value
    [[nodiscard]] T value_or(T default_value) const& {
        if (has_value()) {
            return value();
        }
        return default_value;
    }

    /// @brief Get value or default (rvalue)
    [[nodiscard]] T value_or(T default_value) && {
        if (has_value()) {
            return std::move(*this).value();
        }
        return default_value;
    }

    /// @brief Dereference operator
    /// @return Reference to the value
    /// @pre has_value() must be true
    [[nodiscard]] T& operator*() & { return value(); }

    /// @brief Dereference operator (const)
    [[nodiscard]] const T& operator*() const& { return value(); }

    /// @brief Dereference operator (rvalue)
    [[nodiscard]] T&& operator*() && { return std::move(*this).value(); }

    /// @brief Arrow operator
    /// @return Pointer to the value
    /// @pre has_value() must be true
    [[nodiscard]] T* operator->() { return &value(); }

    /// @brief Arrow operator (const)
    [[nodiscard]] const T* operator->() const { return &value(); }

    // -------------------------------------------------------------------------
    // Monadic Operations
    // -------------------------------------------------------------------------

    /// @brief Transform the value if present
    /// @tparam F Function type T -> U
    /// @param f Function to apply to value
    /// @return `result<U>` with transformed value or original error
    template <typename F>
        requires std::invocable<F, T>
    [[nodiscard]] auto map(F&& f) const& -> result<std::invoke_result_t<F, T>> {
        using U = std::invoke_result_t<F, T>;
        if (has_value()) {
            return result<U>{std::invoke(std::forward<F>(f), value())};
        }
        return result<U>{error()};
    }

    /// @brief Transform the value if present (rvalue)
    template <typename F>
        requires std::invocable<F, T>
    [[nodiscard]] auto map(F&& f) && -> result<std::invoke_result_t<F, T>> {
        using U = std::invoke_result_t<F, T>;
        if (has_value()) {
            return result<U>{std::invoke(std::forward<F>(f), std::move(*this).value())};
        }
        return result<U>{error()};
    }

    /// @brief Chain operations that return result
    /// @tparam F Function type T -> `result<U>`
    /// @param f Function to apply to value
    /// @return `result<U>` from f, or original error if this is error
    template <typename F>
        requires std::invocable<F, T>
    [[nodiscard]] auto and_then(F&& f) const& -> std::invoke_result_t<F, T> {
        using ResultU = std::invoke_result_t<F, T>;
        if (has_value()) {
            return std::invoke(std::forward<F>(f), value());
        }
        return ResultU{error()};
    }

    /// @brief Chain operations that return result (rvalue)
    template <typename F>
        requires std::invocable<F, T>
    [[nodiscard]] auto and_then(F&& f) && -> std::invoke_result_t<F, T> {
        using ResultU = std::invoke_result_t<F, T>;
        if (has_value()) {
            return std::invoke(std::forward<F>(f), std::move(*this).value());
        }
        return ResultU{error()};
    }

    /// @brief Transform error if present
    /// @tparam F Function type status -> status
    /// @param f Function to transform error
    /// @return result with original value or transformed error
    template <typename F>
        requires std::invocable<F, status>
    [[nodiscard]] result transform_error(F&& f) const& {
        if (has_error()) {
            return result{std::invoke(std::forward<F>(f), error())};
        }
        return *this;
    }

    /// @brief Transform error if present (deprecated alias)
    /// @deprecated Use transform_error() instead
    template <typename F>
        requires std::invocable<F, status>
    [[deprecated("use transform_error()")]]
    [[nodiscard]] result map_error(F&& f) const& {
        return transform_error(std::forward<F>(f));
    }

    /// @brief Provide alternative value on error
    /// @tparam F Function type status -> T
    /// @param f Function to produce alternative value
    /// @return Value if success, or result of f(error) if error
    template <typename F>
        requires std::invocable<F, status> && std::convertible_to<std::invoke_result_t<F, status>, T>
    [[nodiscard]] T or_else(F&& f) const& {
        if (has_value()) {
            return value();
        }
        return std::invoke(std::forward<F>(f), error());
    }

    // -------------------------------------------------------------------------
    // Static Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create a success result
    /// @param value The success value
    /// @return result containing the value
    template <typename U = T>
        requires std::constructible_from<T, U&&>
    [[nodiscard]] static result success(U&& value) {
        return result{std::in_place, std::forward<U>(value)};
    }

    /// @brief Create a success result with in-place construction
    /// @tparam Args Constructor argument types
    /// @param args Arguments to forward to T's constructor
    /// @return result containing the constructed value
    template <typename... Args>
        requires std::constructible_from<T, Args...>
    [[nodiscard]] static result success_in_place(Args&&... args) {
        return result{std::in_place, std::forward<Args>(args)...};
    }

    /// @brief Create a failure result
    /// @param error The error status
    /// @return result containing the error
    [[nodiscard]] static result failure(status error) {
        return result{std::move(error)};
    }

private:
    std::variant<T, status> data_;
};

// =============================================================================
// Result<void> Specialization
// =============================================================================

/// @brief Specialization of result for void (operation success/failure only)
/// @details Used for operations that don't return a value but may fail.
template <>
class result<void> {
public:
    using value_type = void;
    using error_type = status;

    /// @brief Default constructor (success)
    /// @note Not constexpr because status contains std::string which is not constexpr in GCC 11
    result() noexcept = default;

    /// @brief Construct with error status
    /// @param error The error status
    result(status error) {
        DTL_ASSERT(error.is_error());  // Check BEFORE move
        error_ = std::move(error);
    }

    /// @brief Construct with error status code
    /// @param code The error status code
    result(status_code code) : error_{status{code}} {
        DTL_ASSERT(code != status_code::ok);
    }

    /// @brief Construct from error object
    /// @param err The error object
    result(const class error& err) : error_{err.get_status()} {}

    /// @brief Construct from error object (move)
    /// @param err The error object
    result(class error&& err) : error_{err.get_status()} {}

    /// @brief Check if result is success
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return error_.ok();
    }

    /// @brief Check if result is error
    [[nodiscard]] constexpr bool has_error() const noexcept {
        return error_.is_error();
    }

    /// @brief Bool conversion (true if success)
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return has_value();
    }

    /// @brief Get the error status
    [[nodiscard]] const status& error() const& {
        return error_;
    }

    /// @brief Transform to value-returning result on success
    template <typename F>
        requires std::invocable<F>
    [[nodiscard]] auto map(F&& f) const -> result<std::invoke_result_t<F>> {
        using U = std::invoke_result_t<F>;
        if (has_value()) {
            return result<U>{std::invoke(std::forward<F>(f))};
        }
        return result<U>{error_};
    }

    /// @brief Chain operations that return result
    template <typename F>
        requires std::invocable<F>
    [[nodiscard]] auto and_then(F&& f) const -> std::invoke_result_t<F> {
        using ResultU = std::invoke_result_t<F>;
        if (has_value()) {
            return std::invoke(std::forward<F>(f));
        }
        return ResultU{error_};
    }

    /// @brief Transform error if present
    /// @tparam F Function type status -> status
    /// @param f Function to transform error
    /// @return result with original value or transformed error
    template <typename F>
        requires std::invocable<F, status>
    [[nodiscard]] result transform_error(F&& f) const {
        if (has_error()) {
            return result{std::invoke(std::forward<F>(f), error_)};
        }
        return *this;
    }

    /// @brief Transform error if present (deprecated alias)
    /// @deprecated Use transform_error() instead
    template <typename F>
        requires std::invocable<F, status>
    [[deprecated("use transform_error()")]]
    [[nodiscard]] result map_error(F&& f) const {
        return transform_error(std::forward<F>(f));
    }

    /// @brief Provide alternative on error
    /// @tparam F Function type status -> void
    /// @param f Function to execute on error (for side effects)
    template <typename F>
        requires std::invocable<F, status>
    void or_else(F&& f) const {
        if (has_error()) {
            std::invoke(std::forward<F>(f), error_);
        }
    }

    // -------------------------------------------------------------------------
    // Static Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create a success result (void)
    /// @return result<void> representing success
    [[nodiscard]] static result success() {
        return result{};
    }

    /// @brief Create a failure result
    /// @param error The error status
    /// @return result<void> containing the error
    [[nodiscard]] static result failure(status error) {
        return result{std::move(error)};
    }

private:
    status error_;
};

// =============================================================================
// Factory Functions
// =============================================================================

/// @brief Create a success result with a value
/// @tparam T The value type
/// @param value The success value
/// @return result<T> containing the value
template <typename T>
[[nodiscard]] inline result<std::decay_t<T>> make_result(T&& value) {
    return result<std::decay_t<T>>{std::forward<T>(value)};
}

/// @brief Create a success result<void>
/// @return result<void> representing success
[[nodiscard]] inline result<void> make_ok_result() {
    return result<void>{};
}

/// @brief Create an error result
/// @tparam T The value type for the result
/// @param error The error status
/// @return result<T> containing the error
template <typename T>
[[nodiscard]] inline result<T> make_error_result(status error) {
    return result<T>{std::move(error)};
}

/// @brief Create an error result from status code
/// @tparam T The value type for the result
/// @param code The error status code
/// @return result<T> containing the error
template <typename T>
[[nodiscard]] inline result<T> make_error_result(status_code code) {
    return result<T>{code};
}

/// @brief Create a result<T> containing an error with message
/// @tparam T The result value type
/// @param code The status code
/// @param message Error message
/// @param loc Source location (auto-captured)
/// @return result<T> containing the error
template <typename T>
[[nodiscard]] inline result<T> make_error(
    status_code code,
    std::string message,
    std::source_location loc = std::source_location::current()) {
    return result<T>{error{code, no_rank, std::move(message), loc}};
}

/// @brief Create a result<T> containing an error with rank and message
/// @tparam T The result value type
/// @param code The status code
/// @param rank Rank where error occurred
/// @param message Error message
/// @param loc Source location (auto-captured)
/// @return result<T> containing the error
template <typename T>
[[nodiscard]] inline result<T> make_error(
    status_code code,
    rank_t rank,
    std::string message,
    std::source_location loc = std::source_location::current()) {
    return result<T>{error{code, rank, std::move(message), loc}};
}

}  // namespace dtl
