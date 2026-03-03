// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_lifecycle.hpp
/// @brief MPI lifecycle query and management helpers
/// @details Provides low-level functions for querying and controlling the MPI
///          runtime state. Used by the environment class to implement ownership
///          modes (dtl_owns, adopt_external, etc.).
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <string>
#include <string_view>

namespace dtl {
namespace mpi {

// =============================================================================
// MPI State Enumeration
// =============================================================================

/// @brief Represents the observable state of the MPI runtime
enum class mpi_state {
    /// MPI has not been initialized
    not_initialized,

    /// MPI is currently initialized and usable
    initialized,

    /// MPI has been finalized and cannot be restarted
    finalized
};

/// @brief Convert mpi_state to human-readable string
/// @param state The MPI state
/// @return String view of the state name
[[nodiscard]] constexpr std::string_view to_string(mpi_state state) noexcept {
    switch (state) {
        case mpi_state::not_initialized: return "not_initialized";
        case mpi_state::initialized:     return "initialized";
        case mpi_state::finalized:       return "finalized";
        default:                         return "unknown";
    }
}

// =============================================================================
// MPI Thread Level Constants
// =============================================================================

/// @brief MPI thread support level constants
/// @details Mirror MPI_THREAD_* values for use without MPI headers.
struct thread_levels {
    static constexpr int single     = 0;  ///< MPI_THREAD_SINGLE
    static constexpr int funneled   = 1;  ///< MPI_THREAD_FUNNELED
    static constexpr int serialized = 2;  ///< MPI_THREAD_SERIALIZED
    static constexpr int multiple   = 3;  ///< MPI_THREAD_MULTIPLE
};

// =============================================================================
// MPI Lifecycle Query Functions
// =============================================================================

/// @brief Check if MPI has been initialized
/// @return true if MPI_Init or MPI_Init_thread has been called
[[nodiscard]] inline bool is_initialized() noexcept {
#if DTL_ENABLE_MPI
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
#else
    return false;
#endif
}

/// @brief Check if MPI has been finalized
/// @return true if MPI_Finalize has been called
[[nodiscard]] inline bool is_finalized() noexcept {
#if DTL_ENABLE_MPI
    int flag = 0;
    MPI_Finalized(&flag);
    return flag != 0;
#else
    return false;
#endif
}

/// @brief Query the thread support level provided by MPI
/// @return result<int> containing the thread level, or error if MPI is
///         not available or not initialized
[[nodiscard]] inline result<int> query_thread_level() {
#if DTL_ENABLE_MPI
    if (!is_initialized()) {
        return make_error<int>(status_code::invalid_state,
                               "MPI is not initialized; cannot query thread level");
    }
    int provided = 0;
    int rc = MPI_Query_thread(&provided);
    if (rc != MPI_SUCCESS) {
        return make_error<int>(status_code::mpi_error,
                               "MPI_Query_thread failed");
    }
    return result<int>{provided};
#else
    return make_error<int>(status_code::not_supported,
                           "MPI support not enabled (DTL_ENABLE_MPI not defined)");
#endif
}

/// @brief Get the current observable MPI state
/// @return mpi_state reflecting the current MPI runtime condition
[[nodiscard]] inline mpi_state get_state() noexcept {
#if DTL_ENABLE_MPI
    // Check finalized first: once finalized, MPI_Initialized may still
    // return true on some implementations.
    if (is_finalized()) {
        return mpi_state::finalized;
    }
    if (is_initialized()) {
        return mpi_state::initialized;
    }
    return mpi_state::not_initialized;
#else
    return mpi_state::not_initialized;
#endif
}

// =============================================================================
// MPI Lifecycle Management Functions
// =============================================================================

/// @brief Initialize MPI with thread support
/// @param required_level Desired thread support level (0-3)
/// @param allow_fallback If true, accept a lower thread level than requested
/// @return result<int> containing the provided thread level on success,
///         or error if initialization fails
[[nodiscard]] inline result<int> initialize(int* argc, char*** argv,
                                            int required_level, bool allow_fallback) {
#if DTL_ENABLE_MPI
    if (is_finalized()) {
        return make_error<int>(status_code::invalid_state,
                               "MPI has been finalized and cannot be reinitialized");
    }

    // If MPI is already initialized, adopt it instead of erroring
    if (is_initialized()) {
        auto level_res = query_thread_level();
        int provided = level_res.has_value() ? level_res.value() : -1;

        // Check if existing thread level is sufficient
        if (!allow_fallback && provided >= 0 && provided < required_level) {
            return make_error<int>(status_code::backend_init_failed,
                                   "MPI already initialized with thread level " +
                                   std::to_string(provided) +
                                   " but required " +
                                   std::to_string(required_level));
        }

        return result<int>{provided};
    }

    int provided = 0;
    int rc = MPI_Init_thread(argc, argv, required_level, &provided);
    if (rc != MPI_SUCCESS) {
        return make_error<int>(status_code::backend_init_failed,
                               "MPI_Init_thread failed with error code " +
                               std::to_string(rc));
    }

    // Check if provided level is sufficient
    if (!allow_fallback && provided < required_level) {
        // Roll back: finalize MPI since we cannot meet the requirement
        MPI_Finalize();
        return make_error<int>(status_code::backend_init_failed,
                               "MPI provided thread level " +
                               std::to_string(provided) +
                               " but required " +
                               std::to_string(required_level) +
                               " (fallback disabled)");
    }

    return result<int>{provided};
#else
    (void)argc;
    (void)argv;
    (void)required_level;
    (void)allow_fallback;
    return make_error<int>(status_code::not_supported,
                           "MPI support not enabled (DTL_ENABLE_MPI not defined)");
#endif
}

/// @brief Initialize MPI with nullptr argc/argv (legacy overload)
/// @param required_level The MPI thread level to request
/// @param allow_fallback If false, fail if provided < required
/// @return result<int> containing the provided thread level on success
/// @note Some MPI implementations may not work correctly with nullptr argc/argv
[[nodiscard]] inline result<int> initialize(int required_level, bool allow_fallback) {
    // Some MPI implementations do not accept nullptr argc/argv.
    // Provide stable dummy storage so MPI_Init_thread always receives
    // valid pointers.
    static int dummy_argc = 0;
    static char** dummy_argv = nullptr;
    return initialize(&dummy_argc, &dummy_argv, required_level, allow_fallback);
}

/// @brief Verify that MPI is already initialized (for adopt mode)
/// @return result<void> success if MPI is initialized, error otherwise
[[nodiscard]] inline result<void> verify_initialized() {
#if DTL_ENABLE_MPI
    if (is_finalized()) {
        return make_error<void>(status_code::invalid_state,
                                "MPI has been finalized; cannot adopt");
    }
    if (!is_initialized()) {
        return make_error<void>(status_code::invalid_state,
                                "MPI is not initialized; cannot adopt external MPI");
    }
    return result<void>{};
#else
    return make_error<void>(status_code::not_supported,
                            "MPI support not enabled (DTL_ENABLE_MPI not defined)");
#endif
}

/// @brief Finalize the MPI runtime
/// @return result<void> success if finalization succeeds, error otherwise
[[nodiscard]] inline result<void> finalize_mpi() {
#if DTL_ENABLE_MPI
    if (!is_initialized()) {
        return make_error<void>(status_code::invalid_state,
                                "MPI is not initialized; cannot finalize");
    }
    if (is_finalized()) {
        return make_error<void>(status_code::invalid_state,
                                "MPI has already been finalized");
    }

    int rc = MPI_Finalize();
    if (rc != MPI_SUCCESS) {
        return make_error<void>(status_code::mpi_error,
                                "MPI_Finalize failed with error code " +
                                std::to_string(rc));
    }
    return result<void>{};
#else
    return make_error<void>(status_code::not_supported,
                            "MPI support not enabled (DTL_ENABLE_MPI not defined)");
#endif
}

}  // namespace mpi
}  // namespace dtl
