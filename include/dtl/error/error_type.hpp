// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file error_type.hpp
/// @brief Error structure with code, rank, and message
/// @details Extended error information including source location and context.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/status.hpp>

#include <source_location>
#include <string>
#include <string_view>

namespace dtl {

/// @brief Extended error information with context
/// @details Provides detailed error information including:
///          - Status code and message
///          - Rank where error occurred
///          - Source location (file, line, function)
///          - Optional additional context
class error {
public:
    /// @brief Construct an error from status
    /// @param stat The status representing the error
    /// @param loc Source location where error was created
    explicit error(
        status stat,
        std::source_location loc = std::source_location::current())
        : status_{std::move(stat)}, location_{loc} {}

    /// @brief Construct an error from status code
    /// @param code The status code
    /// @param rank The rank where error occurred
    /// @param message Optional error message
    /// @param loc Source location where error was created
    error(status_code code,
          rank_t rank = no_rank,
          std::string message = {},
          std::source_location loc = std::source_location::current())
        : status_{code, rank, std::move(message)}, location_{loc} {}

    /// @brief Get the underlying status
    [[nodiscard]] const status& get_status() const noexcept { return status_; }

    /// @brief Get the status code
    [[nodiscard]] status_code code() const noexcept { return status_.code(); }

    /// @brief Get the rank where error occurred
    [[nodiscard]] rank_t rank() const noexcept { return status_.rank(); }

    /// @brief Get the error message
    [[nodiscard]] const std::string& message() const noexcept {
        return status_.message();
    }

    /// @brief Get the source location
    [[nodiscard]] const std::source_location& location() const noexcept {
        return location_;
    }

    /// @brief Get file name where error occurred
    [[nodiscard]] const char* file_name() const noexcept {
        return location_.file_name();
    }

    /// @brief Get line number where error occurred
    [[nodiscard]] std::uint_least32_t line() const noexcept {
        return location_.line();
    }

    /// @brief Get function name where error occurred
    [[nodiscard]] const char* function_name() const noexcept {
        return location_.function_name();
    }

    /// @brief Convert to human-readable string with full context
    [[nodiscard]] std::string to_string() const {
        std::string result = status_.to_string();
        result += " at ";
        result += location_.file_name();
        result += ":";
        result += std::to_string(location_.line());
        result += " in ";
        result += location_.function_name();
        return result;
    }

    /// @brief Implicit conversion to status
    [[nodiscard]] operator const status&() const noexcept { return status_; }

private:
    status status_;
    std::source_location location_;
};

/// @brief Create an error with message (convenience overload)
/// @param code The status code
/// @param message Error message
/// @param loc Source location (auto-captured)
/// @return Error with captured source location
[[nodiscard]] inline error make_error(
    status_code code,
    std::string message,
    std::source_location loc = std::source_location::current()) {
    return error{code, no_rank, std::move(message), loc};
}

/// @brief Create an error with rank and message
/// @param code The status code
/// @param rank Rank where error occurred
/// @param message Error message
/// @param loc Source location (auto-captured)
/// @return Error with captured source location
[[nodiscard]] inline error make_error(
    status_code code,
    rank_t rank,
    std::string message,
    std::source_location loc = std::source_location::current()) {
    return error{code, rank, std::move(message), loc};
}

/// @brief Create an error with just a code
/// @param code The status code
/// @param loc Source location (auto-captured)
/// @return Error with captured source location
[[nodiscard]] inline error make_error(
    status_code code,
    std::source_location loc = std::source_location::current()) {
    return error{code, no_rank, {}, loc};
}

}  // namespace dtl
