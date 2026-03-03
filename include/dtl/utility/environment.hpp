// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file environment.hpp
/// @brief MPI/backend environment RAII wrapper
/// @details Provides RAII management for backend initialization and finalization.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dtl {
namespace legacy {

// ============================================================================
// Environment State
// ============================================================================

/// @brief Current state of the backend environment
enum class environment_state {
    /// @brief Environment not initialized
    uninitialized,

    /// @brief Environment is initializing
    initializing,

    /// @brief Environment is fully initialized and ready
    initialized,

    /// @brief Environment is being finalized
    finalizing,

    /// @brief Environment has been finalized
    finalized,

    /// @brief Environment is in an error state
    error
};

/// @brief Get string representation of environment state
[[nodiscard]] constexpr std::string_view to_string(environment_state state) noexcept {
    switch (state) {
        case environment_state::uninitialized: return "uninitialized";
        case environment_state::initializing: return "initializing";
        case environment_state::initialized: return "initialized";
        case environment_state::finalizing: return "finalizing";
        case environment_state::finalized: return "finalized";
        case environment_state::error: return "error";
        default: return "unknown";
    }
}

// ============================================================================
// Environment Configuration
// ============================================================================

/// @brief Configuration options for environment initialization
struct environment_config {
    /// @brief Enable thread support (MPI_THREAD_MULTIPLE)
    bool thread_support = false;

    /// @brief Required thread support level (MPI-style)
    /// 0=SINGLE, 1=FUNNELED, 2=SERIALIZED, 3=MULTIPLE
    int required_thread_level = 0;

    /// @brief Enable CUDA/GPU backend if available
    bool enable_gpu = false;

    /// @brief Specific GPU device ID to use (-1 = auto-select)
    int gpu_device_id = -1;

    /// @brief Enable verbose initialization output
    bool verbose = false;

    /// @brief Custom error handler for initialization errors
    std::function<void(const std::string&)> error_handler;
};

// ============================================================================
// Environment Info
// ============================================================================

/// @brief Information about the initialized environment
struct environment_info {
    /// @brief Current state
    environment_state state = environment_state::uninitialized;

    /// @brief This process's rank in world communicator
    rank_t world_rank = no_rank;

    /// @brief Total number of ranks in world communicator
    rank_t world_size = 0;

    /// @brief Provided thread support level
    int provided_thread_level = 0;

    /// @brief Whether GPU backend is available
    bool gpu_available = false;

    /// @brief GPU device ID in use (-1 = none)
    int gpu_device_id = -1;

    /// @brief Backend name (e.g., "MPI", "SHMEM", "Single")
    std::string backend_name;

    /// @brief Backend version string
    std::string backend_version;

    /// @brief Processor/node name
    std::string processor_name;
};

// ============================================================================
// Environment Base Class
// ============================================================================

/// @brief Base class for backend environments
/// @details Provides common interface for different backend implementations.
///          Subclasses implement actual MPI, SHMEM, or other backends.
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] environment_base {
public:
    /// @brief Virtual destructor
    virtual ~environment_base() = default;

    /// @brief Initialize the environment
    /// @param argc Argument count from main
    /// @param argv Argument vector from main
    /// @param config Configuration options
    /// @return Success or error
    virtual result<void> initialize(int& argc, char**& argv,
                                    const environment_config& config) = 0;

    /// @brief Finalize the environment
    /// @return Success or error
    virtual result<void> finalize() = 0;

    /// @brief Get current environment state
    [[nodiscard]] virtual environment_state state() const noexcept = 0;

    /// @brief Get environment info
    [[nodiscard]] virtual const environment_info& info() const noexcept = 0;

    /// @brief Check if environment is initialized
    [[nodiscard]] virtual bool initialized() const noexcept = 0;

    /// @brief Abort all processes
    /// @param error_code Error code to return
    /// @param message Optional error message
    [[noreturn]] virtual void abort(int error_code, const std::string& message = "") = 0;

protected:
    environment_info info_;
};

// ============================================================================
// Single-Process Environment
// ============================================================================

/// @brief Single-process (non-distributed) environment
/// @details Provides a valid environment for single-process execution,
///          useful for testing and development without MPI.
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] single_environment : public environment_base {
public:
    /// @brief Default constructor
    single_environment() = default;

    /// @brief Destructor (calls finalize if needed)
    ~single_environment() override {
        if (state() == environment_state::initialized) {
            finalize();
        }
    }

    // Non-copyable
    single_environment(const single_environment&) = delete;
    single_environment& operator=(const single_environment&) = delete;

    // Movable
    single_environment(single_environment&&) = default;
    single_environment& operator=(single_environment&&) = default;

    /// @brief Initialize single-process environment
    result<void> initialize(int& argc, char**& argv,
                           const environment_config& config) override {
        (void)argc;
        (void)argv;
        (void)config;

        info_.state = environment_state::initializing;
        info_.world_rank = 0;
        info_.world_size = 1;
        info_.provided_thread_level = config.required_thread_level;
        info_.gpu_available = config.enable_gpu;
        info_.gpu_device_id = config.enable_gpu ? 0 : -1;
        info_.backend_name = "Single";
        info_.backend_version = "1.0";
        info_.processor_name = "localhost";
        info_.state = environment_state::initialized;

        return {};
    }

    /// @brief Finalize single-process environment
    result<void> finalize() override {
        if (info_.state != environment_state::initialized) {
            return make_error<void>(status_code::invalid_state,
                                    "Environment not initialized");
        }
        info_.state = environment_state::finalizing;
        info_.state = environment_state::finalized;
        return {};
    }

    /// @brief Get current state
    [[nodiscard]] environment_state state() const noexcept override {
        return info_.state;
    }

    /// @brief Get environment info
    [[nodiscard]] const environment_info& info() const noexcept override {
        return info_;
    }

    /// @brief Check if initialized
    [[nodiscard]] bool initialized() const noexcept override {
        return info_.state == environment_state::initialized;
    }

    /// @brief Abort (calls std::abort for single process)
    [[noreturn]] void abort(int error_code, const std::string& message) override {
        if (!message.empty()) {
            std::fprintf(stderr, "DTL Abort: %s (code: %d)\n", message.c_str(), error_code);
        }
        std::abort();
    }
};

// ============================================================================
// RAII Environment Guard
// ============================================================================

/// @brief RAII wrapper for environment lifetime management
/// @tparam Env Environment type
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
template <typename Env = single_environment>
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] scoped_environment {
public:
    /// @brief Construct and initialize environment
    /// @param argc Argument count from main
    /// @param argv Argument vector from main
    /// @param config Configuration options
    scoped_environment(int& argc, char**& argv,
                       const environment_config& config = {}) {
        auto result = env_.initialize(argc, argv, config);
        if (!result) {
            throw std::runtime_error("Failed to initialize environment: " +
                                     std::string(result.error().message()));
        }
    }

    /// @brief Destructor (finalizes environment)
    ~scoped_environment() {
        if (env_.initialized()) {
            env_.finalize();
        }
    }

    // Non-copyable, non-movable (manages global state)
    scoped_environment(const scoped_environment&) = delete;
    scoped_environment& operator=(const scoped_environment&) = delete;
    scoped_environment(scoped_environment&&) = delete;
    scoped_environment& operator=(scoped_environment&&) = delete;

    /// @brief Get the underlying environment
    [[nodiscard]] Env& get() noexcept { return env_; }

    /// @brief Get the underlying environment (const)
    [[nodiscard]] const Env& get() const noexcept { return env_; }

    /// @brief Arrow operator for environment access
    [[nodiscard]] Env* operator->() noexcept { return &env_; }

    /// @brief Arrow operator for environment access (const)
    [[nodiscard]] const Env* operator->() const noexcept { return &env_; }

    /// @brief Get environment info
    [[nodiscard]] const environment_info& info() const noexcept {
        return env_.info();
    }

    /// @brief Get world rank
    [[nodiscard]] rank_t rank() const noexcept { return env_.info().world_rank; }

    /// @brief Get world size
    [[nodiscard]] rank_t size() const noexcept { return env_.info().world_size; }

    /// @brief Check if this is rank 0
    [[nodiscard]] bool is_root() const noexcept { return rank() == 0; }

private:
    Env env_;
};

// NOTE: The global environment accessor layer (detail::g_environment,
// global_environment(), environment_initialized(), world_rank(),
// world_size(), is_root()) was removed as dead code. The g_environment
// pointer was only set by the deprecated mpi_environment, and the
// accessor functions always returned nullptr/false/no_rank/0 in
// non-MPI builds. Use the dtl::environment singleton for environment
// state queries.

// NOTE: register_init_callback() and register_finalize_callback() were
// removed because they were declared but never defined (linker bombs).
// The new dtl::environment singleton manages init/finalize
// through RAII scoping, not callbacks.

}  // namespace legacy
}  // namespace dtl

// ============================================================================
// Backward Compatibility Note (Phase 12.5 / V1.2.1)
// ============================================================================
// The modern dtl::environment RAII singleton is available via:
//   #include <dtl/core/environment.hpp>
// This header (utility/environment.hpp) retains the original
// environment_base / single_environment / scoped_environment classes
// in the dtl::legacy:: namespace (deprecated).
