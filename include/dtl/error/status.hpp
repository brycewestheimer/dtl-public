// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file status.hpp
/// @brief Status codes and status class for operation results
/// @details Defines status_code enumeration with categorized error codes
///          and the status class for representing operation outcomes.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <string>
#include <string_view>

namespace dtl {

// =============================================================================
// Status Code Enumeration
// =============================================================================

/// @brief Enumeration of status codes for DTL operations
/// @details Status codes are organized by category:
///          - 0: Success
///          - 100-199: Communication errors
///          - 200-299: Memory errors
///          - 300-399: Serialization errors
///          - 400-499: Index/bounds errors
///          - 500-599: Backend errors
///          - 600-699: Algorithm errors
///          - 900-999: Internal/unknown errors
enum class status_code : int {
    // Success
    ok = 0,                              ///< Operation completed successfully

    // Non-error sentinel values (1-99)
    not_found = 1,                       ///< Key/element not found (non-error sentinel)
    end_iterator = 2,                    ///< Iterator past-the-end (non-error sentinel)

    // Communication errors (100-199)
    communication_error = 100,           ///< Generic communication failure
    send_failed = 101,                   ///< Send operation failed
    recv_failed = 102,                   ///< Receive operation failed
    broadcast_failed = 103,              ///< Broadcast operation failed
    reduce_failed = 104,                 ///< Reduce operation failed
    barrier_failed = 105,                ///< Barrier synchronization failed
    timeout = 106,                       ///< Operation timed out
    canceled = 107,                      ///< Operation was canceled
    connection_lost = 108,               ///< Connection to peer lost
    rank_failure = 109,                  ///< Remote rank has failed
    collective_failure = 110,            ///< Collective operation failed
    collective_participation_error = 111,///< Not all required ranks participated in collective

    // Memory errors (200-299)
    memory_error = 200,                  ///< Generic memory error
    allocation_failed = 201,             ///< Memory allocation failed
    out_of_memory = 202,                 ///< Insufficient memory available
    invalid_pointer = 203,               ///< Invalid memory pointer
    memory_transfer_failed = 204,        ///< Host-device transfer failed
    device_memory_error = 205,           ///< GPU memory error

    // Serialization errors (300-399)
    serialization_error = 300,           ///< Generic serialization error
    serialize_failed = 301,              ///< Failed to serialize data
    deserialize_failed = 302,            ///< Failed to deserialize data
    buffer_too_small = 303,              ///< Provided buffer too small
    invalid_format = 304,                ///< Invalid serialization format

    // Index/bounds errors (400-499)
    bounds_error = 400,                  ///< Generic bounds error
    out_of_bounds = 401,                 ///< Index out of valid range
    invalid_index = 402,                 ///< Invalid index value
    invalid_rank = 403,                  ///< Invalid rank identifier
    dimension_mismatch = 404,            ///< Dimension count mismatch
    extent_mismatch = 405,               ///< Extent size mismatch
    key_not_found = 406,                 ///< Key not found in container
    out_of_range = 407,                  ///< Value out of valid range
    invalid_argument = 410,              ///< Invalid argument provided
    null_pointer = 411,                  ///< Null pointer passed where not allowed
    not_supported = 420,                 ///< Operation not supported

    // Backend errors (500-599)
    backend_error = 500,                 ///< Generic backend error
    backend_not_available = 501,         ///< Requested backend unavailable
    backend_init_failed = 502,           ///< Backend initialization failed
    backend_invalid = 503,               ///< Backend is invalid for requested operation
    cuda_error = 510,                    ///< CUDA-specific error
    hip_error = 520,                     ///< HIP-specific error
    mpi_error = 530,                     ///< MPI-specific error
    nccl_error = 540,                    ///< NCCL-specific error
    shmem_error = 550,                   ///< SHMEM-specific error

    // Algorithm errors (600-699)
    algorithm_error = 600,               ///< Generic algorithm error
    operation_failed = 600,              ///< Alias for algorithm_error
    precondition_failed = 601,           ///< Algorithm precondition not met
    postcondition_failed = 602,          ///< Algorithm postcondition not met
    convergence_failed = 603,            ///< Iterative algorithm did not converge

    // Consistency errors (700-799)
    consistency_error = 700,             ///< Generic consistency error
    consistency_violation = 701,         ///< Consistency policy violated
    structural_invalidation = 702,       ///< Structure invalidated during operation

    // Internal/unknown errors (900-999)
    internal_error = 900,                ///< Internal DTL error
    not_implemented = 901,               ///< Feature not yet implemented
    invalid_state = 902,                 ///< Object in invalid state
    unknown_error = 999                  ///< Unknown error occurred
};

/// @brief Get the category name for a status code
/// @param code The status code
/// @return String view of the category name
[[nodiscard]] inline constexpr std::string_view status_category(status_code code) noexcept {
    const int val = static_cast<int>(code);
    if (val >= 0 && val < 100) return "success";
    if (val >= 100 && val < 200) return "communication";
    if (val >= 200 && val < 300) return "memory";
    if (val >= 300 && val < 400) return "serialization";
    if (val >= 400 && val < 500) return "bounds";
    if (val >= 500 && val < 600) return "backend";
    if (val >= 600 && val < 700) return "algorithm";
    if (val >= 700 && val < 800) return "consistency";
    return "internal";
}

/// @brief Get the human-readable name for a status code
/// @param code The status code
/// @return String view of the code name
[[nodiscard]] inline constexpr std::string_view status_code_name(status_code code) noexcept {
    switch (code) {
        case status_code::ok: return "ok";
        // Non-error sentinels
        case status_code::not_found: return "not_found";
        case status_code::end_iterator: return "end_iterator";
        // Communication
        case status_code::communication_error: return "communication_error";
        case status_code::send_failed: return "send_failed";
        case status_code::recv_failed: return "recv_failed";
        case status_code::broadcast_failed: return "broadcast_failed";
        case status_code::reduce_failed: return "reduce_failed";
        case status_code::barrier_failed: return "barrier_failed";
        case status_code::timeout: return "timeout";
        case status_code::canceled: return "canceled";
        case status_code::connection_lost: return "connection_lost";
        case status_code::rank_failure: return "rank_failure";
        case status_code::collective_failure: return "collective_failure";
        case status_code::collective_participation_error: return "collective_participation_error";
        // Memory
        case status_code::memory_error: return "memory_error";
        case status_code::allocation_failed: return "allocation_failed";
        case status_code::out_of_memory: return "out_of_memory";
        case status_code::invalid_pointer: return "invalid_pointer";
        case status_code::memory_transfer_failed: return "memory_transfer_failed";
        case status_code::device_memory_error: return "device_memory_error";
        // Serialization
        case status_code::serialization_error: return "serialization_error";
        case status_code::serialize_failed: return "serialize_failed";
        case status_code::deserialize_failed: return "deserialize_failed";
        case status_code::buffer_too_small: return "buffer_too_small";
        case status_code::invalid_format: return "invalid_format";
        // Bounds
        case status_code::bounds_error: return "bounds_error";
        case status_code::out_of_bounds: return "out_of_bounds";
        case status_code::invalid_index: return "invalid_index";
        case status_code::invalid_rank: return "invalid_rank";
        case status_code::dimension_mismatch: return "dimension_mismatch";
        case status_code::extent_mismatch: return "extent_mismatch";
        case status_code::key_not_found: return "key_not_found";
        case status_code::out_of_range: return "out_of_range";
        case status_code::invalid_argument: return "invalid_argument";
        case status_code::null_pointer: return "null_pointer";
        case status_code::not_supported: return "not_supported";
        // Backend
        case status_code::backend_error: return "backend_error";
        case status_code::backend_not_available: return "backend_not_available";
        case status_code::backend_init_failed: return "backend_init_failed";
        case status_code::backend_invalid: return "backend_invalid";
        case status_code::cuda_error: return "cuda_error";
        case status_code::hip_error: return "hip_error";
        case status_code::mpi_error: return "mpi_error";
        case status_code::nccl_error: return "nccl_error";
        case status_code::shmem_error: return "shmem_error";
        // Algorithm
        case status_code::algorithm_error: return "algorithm_error";
        case status_code::precondition_failed: return "precondition_failed";
        case status_code::postcondition_failed: return "postcondition_failed";
        case status_code::convergence_failed: return "convergence_failed";
        // Consistency
        case status_code::consistency_error: return "consistency_error";
        case status_code::consistency_violation: return "consistency_violation";
        case status_code::structural_invalidation: return "structural_invalidation";
        // Internal
        case status_code::internal_error: return "internal_error";
        case status_code::not_implemented: return "not_implemented";
        case status_code::invalid_state: return "invalid_state";
        case status_code::unknown_error: return "unknown_error";
        default: return "unknown";
    }
}

// =============================================================================
// Status Class
// =============================================================================

/// @brief Represents the outcome of a DTL operation
/// @details Combines a status code with optional rank information and message.
///          Provides a more informative alternative to simple error codes.
class status {
public:
    /// @brief Construct a success status
    /// @note Not constexpr because std::string is not constexpr in GCC 11
    status() noexcept = default;

    /// @brief Construct from status code
    /// @param code The status code
    /// @note Not constexpr because std::string member is not constexpr in GCC 11
    explicit status(status_code code) noexcept
        : code_{code} {}

    /// @brief Construct from status code and rank
    /// @param code The status code
    /// @param rank The rank where the error occurred
    /// @note Not constexpr because std::string member is not constexpr in GCC 11
    status(status_code code, rank_t rank) noexcept
        : code_{code}, rank_{rank} {}

    /// @brief Construct from status code, rank, and message
    /// @param code The status code
    /// @param rank The rank where the error occurred
    /// @param message Additional error information
    status(status_code code, rank_t rank, std::string message)
        : code_{code}, rank_{rank}, message_{std::move(message)} {}

    /// @brief Get the status code
    /// @return The status code
    [[nodiscard]] constexpr status_code code() const noexcept { return code_; }

    /// @brief Check if the status represents success
    /// @return true if status code is ok
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code_ == status_code::ok;
    }

    /// @brief Check if the status represents an error
    /// @return true if status code is not ok
    [[nodiscard]] constexpr bool is_error() const noexcept {
        return code_ != status_code::ok;
    }

    /// @brief Check if the status represents success (bool conversion)
    /// @return true if status code is ok
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }

    /// @brief Get the rank where error occurred
    /// @return The rank, or no_rank if not applicable
    [[nodiscard]] constexpr rank_t rank() const noexcept { return rank_; }

    /// @brief Get the error message
    /// @return The error message, or empty string if none
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

    /// @brief Get category name for this status
    /// @return The category name
    [[nodiscard]] constexpr std::string_view category() const noexcept {
        return status_category(code_);
    }

    /// @brief Convert status to human-readable string
    /// @return Formatted status string
    [[nodiscard]] std::string to_string() const {
        if (ok()) {
            return "ok";
        }
        std::string result = "error: ";
        result += category();
        result += " (code=";
        result += std::to_string(static_cast<int>(code_));
        result += ")";
        if (rank_ != no_rank) {
            result += " on rank ";
            result += std::to_string(rank_);
        }
        if (!message_.empty()) {
            result += ": ";
            result += message_;
        }
        return result;
    }

    /// @brief Comparison operators
    /// @{
    [[nodiscard]] constexpr bool operator==(const status& other) const noexcept {
        return code_ == other.code_;
    }

    [[nodiscard]] constexpr bool operator!=(const status& other) const noexcept {
        return code_ != other.code_;
    }

    [[nodiscard]] constexpr bool operator==(status_code code) const noexcept {
        return code_ == code;
    }

    [[nodiscard]] constexpr bool operator!=(status_code code) const noexcept {
        return code_ != code;
    }
    /// @}

private:
    status_code code_ = status_code::ok;
    rank_t rank_ = no_rank;
    std::string message_;
};

/// @brief Factory function for success status
/// @return A status representing success
/// @note Not constexpr because status contains std::string which is not constexpr in GCC 11
[[nodiscard]] inline status ok_status() noexcept {
    return status{status_code::ok};
}

/// @brief Factory function for error status
/// @param code The error code
/// @param rank Optional rank where error occurred
/// @param message Optional error message
/// @return A status representing the error
[[nodiscard]] inline status error_status(status_code code,
                                         rank_t rank = no_rank,
                                         std::string message = {}) {
    return status{code, rank, std::move(message)};
}

}  // namespace dtl
