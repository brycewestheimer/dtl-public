// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file collective_error.hpp
/// @brief Aggregated errors from collective operations
/// @details Handles the case where different ranks may experience different
///          errors during a collective operation.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/status.hpp>

#include <map>
#include <string>
#include <vector>

namespace dtl {

/// @brief Aggregates errors from multiple ranks in a collective operation
/// @details When a collective operation fails on some ranks but succeeds on
///          others, this class collects all error information for diagnosis.
class collective_error {
public:
    /// @brief Error entry for a single rank
    struct rank_error {
        rank_t rank;           ///< The rank that reported the error
        status error_status;   ///< The error status

        /// @brief Check if this rank had an error
        [[nodiscard]] bool has_error() const noexcept {
            return error_status.is_error();
        }
    };

    /// @brief Default constructor (no errors)
    collective_error() = default;

    /// @brief Construct with total rank count
    /// @param num_ranks Total number of ranks in the collective
    explicit collective_error(rank_t num_ranks)
        : num_ranks_{num_ranks} {}

    /// @brief Add an error from a specific rank
    /// @param rank The rank reporting the error
    /// @param error The error status
    void add_error(rank_t rank, status error) {
        if (error.is_error()) {
            errors_.push_back(rank_error{rank, std::move(error)});
        }
    }

    /// @brief Check if any errors occurred
    [[nodiscard]] bool has_errors() const noexcept {
        return !errors_.empty();
    }

    /// @brief Check if all ranks succeeded
    [[nodiscard]] bool all_succeeded() const noexcept {
        return errors_.empty();
    }

    /// @brief Get the number of ranks with errors
    [[nodiscard]] size_type error_count() const noexcept {
        return errors_.size();
    }

    /// @brief Get the total number of ranks
    [[nodiscard]] rank_t num_ranks() const noexcept {
        return num_ranks_;
    }

    /// @brief Get all rank errors
    [[nodiscard]] const std::vector<rank_error>& errors() const noexcept {
        return errors_;
    }

    /// @brief Get the first error (if any)
    /// @return The first error, or ok status if no errors
    [[nodiscard]] status first_error() const {
        if (errors_.empty()) {
            return ok_status();
        }
        return errors_.front().error_status;
    }

    /// @brief Get the most common error status code
    /// @return The most frequently occurring error code, or ok if no errors
    [[nodiscard]] status_code most_common_error() const {
        if (errors_.empty()) {
            return status_code::ok;
        }

        std::map<status_code, size_type> counts;
        for (const auto& e : errors_) {
            ++counts[e.error_status.code()];
        }

        status_code most_common = status_code::ok;
        size_type max_count = 0;
        for (const auto& [code, count] : counts) {
            if (count > max_count) {
                max_count = count;
                most_common = code;
            }
        }
        return most_common;
    }

    /// @brief Create a summary status for the collective operation
    /// @return ok if all succeeded, otherwise an aggregated error status
    [[nodiscard]] status summary() const {
        if (all_succeeded()) {
            return ok_status();
        }
        std::string msg = std::to_string(error_count()) + " of " +
                          std::to_string(num_ranks_) + " ranks failed";
        return error_status(most_common_error(), no_rank, std::move(msg));
    }

    /// @brief Convert to human-readable string
    [[nodiscard]] std::string to_string() const {
        if (all_succeeded()) {
            return "collective operation succeeded on all " +
                   std::to_string(num_ranks_) + " ranks";
        }

        std::string result = "collective operation failed on " +
                             std::to_string(error_count()) + " of " +
                             std::to_string(num_ranks_) + " ranks:\n";

        for (const auto& e : errors_) {
            result += "  rank " + std::to_string(e.rank) + ": " +
                      e.error_status.to_string() + "\n";
        }
        return result;
    }

private:
    rank_t num_ranks_ = 0;
    std::vector<rank_error> errors_;
};

}  // namespace dtl
