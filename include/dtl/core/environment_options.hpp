// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file environment_options.hpp
/// @brief Configuration options for DTL environment initialization
/// @details Defines per-backend ownership modes and aggregated options for
///          controlling how DTL initializes, adopts, or disables each backend.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>

#include <string>
#include <string_view>

namespace dtl {

// =============================================================================
// Thread Support Level (Backend-Agnostic)
// =============================================================================

/// @brief Backend-agnostic thread support level
/// @details Provides a portable enumeration for thread safety guarantees
///          that can be mapped to backend-specific values (e.g., MPI_THREAD_*).
/// @since 0.1.0
enum class thread_support_level {
    /// @brief Single-threaded (no threading support)
    /// @details Corresponds to MPI_THREAD_SINGLE.
    ///          Application must be single-threaded.
    single = 0,

    /// @brief Funneled (main thread only)
    /// @details Corresponds to MPI_THREAD_FUNNELED.
    ///          Only the main thread may make calls to the library.
    funneled = 1,

    /// @brief Serialized (any thread, but serialized)
    /// @details Corresponds to MPI_THREAD_SERIALIZED.
    ///          Any thread may make calls, but only one at a time.
    serialized = 2,

    /// @brief Full multi-threading support
    /// @details Corresponds to MPI_THREAD_MULTIPLE.
    ///          Any thread may make calls at any time.
    multiple = 3
};

/// @brief Convert thread_support_level to human-readable string
/// @param level The thread support level
/// @return String view of the level name
[[nodiscard]] constexpr std::string_view to_string(thread_support_level level) noexcept {
    switch (level) {
        case thread_support_level::single:     return "single";
        case thread_support_level::funneled:   return "funneled";
        case thread_support_level::serialized: return "serialized";
        case thread_support_level::multiple:   return "multiple";
        default:                               return "unknown";
    }
}

/// @brief Convert MPI thread level constant to generic thread_support_level
/// @param mpi_level MPI thread level (0-3, corresponding to MPI_THREAD_* constants)
/// @return Corresponding thread_support_level enum value
/// @details MPI defines: SINGLE=0, FUNNELED=1, SERIALIZED=2, MULTIPLE=3
[[nodiscard]] constexpr thread_support_level from_mpi_thread_level(int mpi_level) noexcept {
    switch (mpi_level) {
        case 0:  return thread_support_level::single;
        case 1:  return thread_support_level::funneled;
        case 2:  return thread_support_level::serialized;
        case 3:  return thread_support_level::multiple;
        default: return thread_support_level::single;  // Conservative fallback
    }
}

/// @brief Convert generic thread_support_level to MPI thread level constant
/// @param level The generic thread support level
/// @return MPI thread level integer (0-3)
[[nodiscard]] constexpr int to_mpi_thread_level(thread_support_level level) noexcept {
    return static_cast<int>(level);
}

// =============================================================================
// Backend Ownership Modes
// =============================================================================

/// @brief Controls how DTL manages a backend's lifecycle
/// @details Each backend (MPI, CUDA, SHMEM) can be independently configured
///          with one of these ownership modes.
enum class backend_ownership {
    /// DTL initializes and finalizes the backend (full ownership)
    dtl_owns,

    /// DTL adopts an externally-initialized backend (user manages lifecycle)
    adopt_external,

    /// DTL initializes if available, silently skips if not
    optional,

    /// Backend is explicitly disabled regardless of availability
    disabled
};

/// @brief Convert backend_ownership to human-readable string
/// @param ownership The ownership mode
/// @return String view of the ownership mode name
[[nodiscard]] constexpr std::string_view to_string(backend_ownership ownership) noexcept {
    switch (ownership) {
        case backend_ownership::dtl_owns:        return "dtl_owns";
        case backend_ownership::adopt_external:  return "adopt_external";
        case backend_ownership::optional:        return "optional";
        case backend_ownership::disabled:        return "disabled";
        default:                                 return "unknown";
    }
}

// =============================================================================
// Determinism Policy Controls (ARC-0010)
// =============================================================================

/// @brief Global execution determinism mode
enum class determinism_mode {
    /// @brief Prefer peak throughput; implementation may use non-deterministic ordering
    throughput = 0,

    /// @brief Enforce deterministic ordering where supported by backend + algorithm
    deterministic = 1,
};

/// @brief Reduction scheduling policy under deterministic mode
enum class reduction_schedule_policy {
    /// @brief Backend/implementation-defined reduction scheduling (default)
    implementation_defined = 0,

    /// @brief Use a fixed, rank-stable reduction schedule where available
    fixed_tree = 1,
};

/// @brief Progress/event ordering policy under deterministic mode
enum class progress_ordering_policy {
    /// @brief Backend/implementation-defined progress ordering (default)
    implementation_defined = 0,

    /// @brief Prefer rank-stable progress ordering where supported
    rank_ordered = 1,
};

[[nodiscard]] constexpr std::string_view to_string(determinism_mode mode) noexcept {
    switch (mode) {
        case determinism_mode::throughput:    return "throughput";
        case determinism_mode::deterministic: return "deterministic";
        default:                              return "unknown";
    }
}

[[nodiscard]] constexpr std::string_view to_string(reduction_schedule_policy policy) noexcept {
    switch (policy) {
        case reduction_schedule_policy::implementation_defined:
            return "implementation_defined";
        case reduction_schedule_policy::fixed_tree:
            return "fixed_tree";
        default:
            return "unknown";
    }
}

[[nodiscard]] constexpr std::string_view to_string(progress_ordering_policy policy) noexcept {
    switch (policy) {
        case progress_ordering_policy::implementation_defined:
            return "implementation_defined";
        case progress_ordering_policy::rank_ordered:
            return "rank_ordered";
        default:
            return "unknown";
    }
}

/// @brief Determinism control block (ARC-0010)
struct determinism_options {
    /// @brief Throughput-first by default for backward compatibility
    determinism_mode mode = determinism_mode::throughput;

    /// @brief Reduction scheduling control in deterministic mode
    reduction_schedule_policy reduction_schedule = reduction_schedule_policy::implementation_defined;

    /// @brief Progress ordering control in deterministic mode
    progress_ordering_policy progress_ordering = progress_ordering_policy::implementation_defined;
};

// =============================================================================
// Per-Backend Option Structs
// =============================================================================

/// @brief Configuration options for the MPI backend
struct mpi_options {
    /// @brief Ownership mode for MPI lifecycle
    backend_ownership ownership = backend_ownership::dtl_owns;

    /// @brief Required MPI thread support level
    /// @details 0=SINGLE, 1=FUNNELED, 2=SERIALIZED, 3=MULTIPLE
    int thread_level = 1;

    /// @brief Allow fallback to a lower thread level if requested is unavailable
    bool allow_thread_fallback = true;

    /// @brief Custom MPI communicator (e.g., cast from MPI_Comm)
    /// @details If non-null, DTL uses this communicator instead of MPI_COMM_WORLD.
    ///          Only meaningful when ownership is adopt_external.
    void* custom_comm = nullptr;
};

/// @brief Configuration options for the CUDA backend
struct cuda_options {
    /// @brief Ownership mode for CUDA lifecycle
    backend_ownership ownership = backend_ownership::optional;

    /// @brief CUDA device ID to use (-1 = auto-select)
    int device_id = -1;

    /// @brief Whether to eagerly create the CUDA context on initialization
    bool eager_context = false;
};

/// @brief Configuration options for the SHMEM (OpenSHMEM) backend
struct shmem_options {
    /// @brief Ownership mode for SHMEM lifecycle
    backend_ownership ownership = backend_ownership::disabled;

    /// @brief Symmetric heap size in bytes (0 = system default)
    size_type heap_size = 0;
};

/// @brief Configuration options for the NCCL backend
/// @details NCCL communicators manage their own lifecycle (created per
///          communicator group). This option controls detection only.
struct nccl_options {
    /// @brief Ownership mode for NCCL detection
    /// @details NCCL does not require global init/finalize — communicators
    ///          are created and destroyed individually. This controls whether
    ///          DTL reports NCCL as available.
    backend_ownership ownership = backend_ownership::optional;
};

/// @brief Configuration options for the HIP/ROCm backend
struct hip_options {
    /// @brief Ownership mode for HIP lifecycle
    backend_ownership ownership = backend_ownership::optional;

    /// @brief HIP device ID to use (-1 = auto-select)
    int device_id = -1;

    /// @brief Whether to eagerly create the HIP context on initialization
    bool eager_context = false;
};

// =============================================================================
// Aggregated Environment Options
// =============================================================================

/// @brief Aggregated configuration for all backends
/// @details Combines per-backend options with global settings. Use the
///          static factory methods for common configurations.
struct environment_options {
    /// @brief MPI backend configuration
    mpi_options mpi{};

    /// @brief CUDA backend configuration
    cuda_options cuda{};

    /// @brief SHMEM backend configuration
    shmem_options shmem{};

    /// @brief NCCL backend configuration
    nccl_options nccl{};

    /// @brief HIP backend configuration
    hip_options hip{};

    /// @brief Enable verbose output during initialization/finalization
    bool verbose = false;

    /// @brief Determinism controls (ARC-0010)
    determinism_options determinism{};

    /// @brief Named domain label for diagnostics (Rule 7 compliance)
    /// @details Used for error messages and debugging to identify which
    ///          environment instance (and thus which library/subsystem)
    ///          generated a particular message. Does not affect communicator
    ///          topology — isolation comes from MPI_Comm_dup.
    /// @since 0.1.0
    std::string domain = "default";

    // -------------------------------------------------------------------------
    // Futures Configuration (added in v1.3.0)
    // -------------------------------------------------------------------------

    /// @brief Futures progress mode
    /// @details Controls how the progress engine advances async operations:
    ///          - "explicit" (default): User must call poll() or wait functions
    ///          - "background": A background thread automatically polls
    enum class futures_progress_mode {
        explicit_mode,  ///< Explicit polling only
        background      ///< Background thread polls automatically
    };

    /// @brief Futures configuration options
    struct futures_options {
        /// @brief Progress mode
        futures_progress_mode progress_mode = futures_progress_mode::explicit_mode;

        /// @brief Default wait timeout in milliseconds (0 = no timeout)
        uint32_t default_timeout_ms = 30000;

        /// @brief CI wait timeout in milliseconds
        uint32_t ci_timeout_ms = 30000;

        /// @brief Enable diagnostic output on timeout
        bool enable_diagnostics = true;

        /// @brief Poll interval in microseconds for background mode
        uint32_t poll_interval_us = 100;
    };

    /// @brief Futures system configuration
    futures_options futures{};

    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create default options (MPI owned, CUDA optional, SHMEM disabled)
    /// @return Environment options with sensible defaults
    [[nodiscard]] static environment_options defaults() {
        return environment_options{};
    }

    /// @brief Create options for adopting an externally-initialized MPI
    /// @details MPI is adopt_external, CUDA is optional, SHMEM is disabled.
    ///          Use when the application or framework has already called MPI_Init.
    /// @return Environment options with MPI in adopt mode
    [[nodiscard]] static environment_options adopt_mpi() {
        environment_options opts{};
        opts.mpi.ownership = backend_ownership::adopt_external;
        return opts;
    }

    /// @brief Create options that only enable MPI (disable CUDA and SHMEM)
    /// @return Environment options with only MPI active
    [[nodiscard]] static environment_options mpi_only() {
        environment_options opts{};
        opts.cuda.ownership = backend_ownership::disabled;
        opts.shmem.ownership = backend_ownership::disabled;
        return opts;
    }

    /// @brief Create minimal options with all backends disabled
    /// @details Useful for single-process, CPU-only execution.
    /// @return Environment options with everything disabled
    [[nodiscard]] static environment_options minimal() {
        environment_options opts{};
        opts.mpi.ownership = backend_ownership::disabled;
        opts.cuda.ownership = backend_ownership::disabled;
        opts.shmem.ownership = backend_ownership::disabled;
        return opts;
    }

    /// @brief Create options with background progress enabled
    /// @details Useful for applications that don't want to manually poll
    /// @return Environment options with background progress
    [[nodiscard]] static environment_options with_background_progress() {
        environment_options opts{};
        opts.futures.progress_mode = futures_progress_mode::background;
        return opts;
    }
};

}  // namespace dtl
