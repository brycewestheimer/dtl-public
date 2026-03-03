// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_environment.hpp
/// @brief MPI_Init/MPI_Finalize RAII wrapper
/// @details Provides RAII management of MPI lifetime.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/utility/environment.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <string>

namespace dtl {
namespace mpi {

// ============================================================================
// MPI Thread Levels
// ============================================================================

/// @brief MPI thread support levels
enum class thread_level {
    single = 0,      ///< MPI_THREAD_SINGLE
    funneled = 1,    ///< MPI_THREAD_FUNNELED
    serialized = 2,  ///< MPI_THREAD_SERIALIZED
    multiple = 3     ///< MPI_THREAD_MULTIPLE
};

/// @brief Convert thread level to string
[[nodiscard]] constexpr std::string_view to_string(thread_level level) noexcept {
    switch (level) {
        case thread_level::single: return "SINGLE";
        case thread_level::funneled: return "FUNNELED";
        case thread_level::serialized: return "SERIALIZED";
        case thread_level::multiple: return "MULTIPLE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// MPI Environment
// ============================================================================

/// @brief MPI environment with RAII lifetime management
/// @details Handles MPI_Init_thread and MPI_Finalize automatically.
///          Only one instance should exist per process.
class mpi_environment : public legacy::environment_base {
public:
    /// @brief Default constructor (does not initialize MPI)
    mpi_environment() = default;

    /// @brief Destructor (finalizes MPI if initialized)
    ~mpi_environment() override {
        if (state() == legacy::environment_state::initialized) {
            finalize();
        }
    }

    // Non-copyable
    mpi_environment(const mpi_environment&) = delete;
    mpi_environment& operator=(const mpi_environment&) = delete;

    // Movable
    mpi_environment(mpi_environment&&) = default;
    mpi_environment& operator=(mpi_environment&&) = default;

    /// @brief Initialize MPI environment
    /// @param argc Argument count from main
    /// @param argv Argument vector from main
    /// @param config Configuration options
    /// @return Success or error
    result<void> initialize(int& argc, char**& argv,
                           const legacy::environment_config& config) override {
#if DTL_ENABLE_MPI
        if (info_.state != legacy::environment_state::uninitialized) {
            return make_error<void>(status_code::invalid_state,
                                    "MPI already initialized");
        }

        info_.state = legacy::environment_state::initializing;

        // Determine required thread level
        int required = config.required_thread_level;
        int provided = 0;

        int result = MPI_Init_thread(&argc, &argv, required, &provided);
        if (result != MPI_SUCCESS) {
            info_.state = legacy::environment_state::error;
            return make_error<void>(status_code::backend_error,
                                    "MPI_Init_thread failed");
        }

        // Store provided thread level
        info_.provided_thread_level = provided;
        provided_level_ = static_cast<thread_level>(provided);

        // Get world rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &info_.world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &info_.world_size);

        // Get processor name
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        info_.processor_name = std::string(processor_name, name_len);

        // Get MPI version
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        info_.backend_name = "MPI";
        info_.backend_version = std::to_string(version) + "." + std::to_string(subversion);

        info_.state = legacy::environment_state::initialized;

        return {};
#else
        (void)argc;
        (void)argv;
        (void)config;
        return make_error<void>(status_code::not_supported,
                                "MPI support not enabled");
#endif
    }

    /// @brief Finalize MPI environment
    /// @return Success or error
    result<void> finalize() override {
#if DTL_ENABLE_MPI
        if (info_.state != legacy::environment_state::initialized) {
            return make_error<void>(status_code::invalid_state,
                                    "MPI not initialized");
        }

        info_.state = legacy::environment_state::finalizing;

        int result = MPI_Finalize();
        if (result != MPI_SUCCESS) {
            info_.state = legacy::environment_state::error;
            return make_error<void>(status_code::backend_error,
                                    "MPI_Finalize failed");
        }

        info_.state = legacy::environment_state::finalized;
        return {};
#else
        return make_error<void>(status_code::not_supported,
                                "MPI support not enabled");
#endif
    }

    /// @brief Get current state
    [[nodiscard]] legacy::environment_state state() const noexcept override {
        return info_.state;
    }

    /// @brief Get environment info
    [[nodiscard]] const legacy::environment_info& info() const noexcept override {
        return info_;
    }

    /// @brief Check if initialized
    [[nodiscard]] bool initialized() const noexcept override {
        return info_.state == legacy::environment_state::initialized;
    }

    /// @brief Abort all MPI processes
    [[noreturn]] void abort(int error_code, const std::string& message) override {
#if DTL_ENABLE_MPI
        if (!message.empty()) {
            std::fprintf(stderr, "[Rank %d] MPI Abort: %s\n",
                        info_.world_rank, message.c_str());
        }
        MPI_Abort(MPI_COMM_WORLD, error_code);
#else
        if (!message.empty()) {
            std::fprintf(stderr, "MPI Abort: %s\n", message.c_str());
        }
#endif
        std::abort();
    }

    // ------------------------------------------------------------------------
    // MPI-Specific Methods
    // ------------------------------------------------------------------------

    /// @brief Get provided thread level
    [[nodiscard]] thread_level provided_thread_level() const noexcept {
        return provided_level_;
    }

    /// @brief Check if query-response thread level is sufficient
    /// @param required Required thread level
    /// @return true if provided >= required
    [[nodiscard]] bool thread_level_sufficient(thread_level required) const noexcept {
        return static_cast<int>(provided_level_) >= static_cast<int>(required);
    }

    /// @brief Get MPI_COMM_WORLD (as raw communicator)
    /// @return MPI_COMM_WORLD or equivalent
#if DTL_ENABLE_MPI
    [[nodiscard]] MPI_Comm world_comm() const noexcept {
        return MPI_COMM_WORLD;
    }
#endif

    /// @brief Perform a global barrier on MPI_COMM_WORLD
    /// @return Success or error
    result<void> world_barrier() {
#if DTL_ENABLE_MPI
        if (!initialized()) {
            return make_error<void>(status_code::invalid_state,
                                    "MPI not initialized");
        }
        int result = MPI_Barrier(MPI_COMM_WORLD);
        if (result != MPI_SUCCESS) {
            return make_error<void>(status_code::backend_error,
                                    "MPI_Barrier failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported,
                                "MPI support not enabled");
#endif
    }

private:
    thread_level provided_level_ = thread_level::single;
};

// ============================================================================
// Convenience Type Alias
// ============================================================================

/// @brief Scoped MPI environment
using scoped_mpi_environment = legacy::scoped_environment<mpi_environment>;

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Check if MPI is currently initialized (global query)
[[nodiscard]] inline bool mpi_initialized() noexcept {
#if DTL_ENABLE_MPI
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
#else
    return false;
#endif
}

/// @brief Check if MPI has been finalized (global query)
[[nodiscard]] inline bool mpi_finalized() noexcept {
#if DTL_ENABLE_MPI
    int flag = 0;
    MPI_Finalized(&flag);
    return flag != 0;
#else
    return false;
#endif
}

}  // namespace mpi
}  // namespace dtl
