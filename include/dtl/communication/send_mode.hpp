// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file send_mode.hpp
/// @brief MPI send mode variants for point-to-point communication
/// @details Defines the send_mode enum representing the four MPI send
///          semantics: standard, synchronous, ready, and buffered.
///          Used by point-to-point communication adapters to select
///          the appropriate MPI send call.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>

#include <string_view>

namespace dtl {

/// @brief MPI send mode variants
/// @details Each mode maps to a distinct MPI send semantic:
///          - standard:    MPI_Send  - implementation chooses buffering
///          - synchronous: MPI_Ssend - rendezvous (blocks until recv posted)
///          - ready:       MPI_Rsend - recv must already be posted (UB otherwise)
///          - buffered:    MPI_Bsend - user-provided buffer for message storage
enum class send_mode {
    standard,      ///< MPI_Send - implementation chooses buffering
    synchronous,   ///< MPI_Ssend - rendezvous (blocks until recv posted)
    ready,         ///< MPI_Rsend - recv must already be posted
    buffered       ///< MPI_Bsend - user-provided buffer
};

/// @brief Convert send_mode to human-readable string
/// @param mode The send mode to convert
/// @return String view of the mode name
[[nodiscard]] constexpr std::string_view to_string(send_mode mode) noexcept {
    switch (mode) {
        case send_mode::standard:    return "standard";
        case send_mode::synchronous: return "synchronous";
        case send_mode::ready:       return "ready";
        case send_mode::buffered:    return "buffered";
        default:                     return "unknown";
    }
}

}  // namespace dtl
