// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file resource.hpp
/// @brief Unified resource initialization and management
/// @details Provides RAII-based environment management and unified initialization
///          pattern for DTL backends.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <memory>
#include <functional>
#include <vector>

namespace dtl {
namespace legacy {

// ============================================================================
// Initialization Options
// ============================================================================

/// @brief Configuration options for DTL initialization
/// @deprecated Use dtl::environment_options from <dtl/core/environment_options.hpp>
struct [[deprecated("Use dtl::environment_options from <dtl/core/environment_options.hpp>")]] init_options {
    /// @brief Thread support level requested
    enum class thread_level {
        single,      ///< MPI_THREAD_SINGLE equivalent
        funneled,    ///< MPI_THREAD_FUNNELED equivalent
        serialized,  ///< MPI_THREAD_SERIALIZED equivalent
        multiple     ///< MPI_THREAD_MULTIPLE equivalent (default)
    };

    /// @brief Requested thread support level
    thread_level threading = thread_level::multiple;

    /// @brief Enable MPI backend (if available)
    bool enable_mpi = true;

    /// @brief Enable CUDA backend (if available)
    bool enable_cuda = true;

    /// @brief Enable shared memory backend (always available)
    bool enable_shared_memory = true;

    /// @brief Verbose initialization logging
    bool verbose = false;
};

// ============================================================================
// Capabilities
// ============================================================================

/// @brief Backend capability information discovered at initialization
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
struct [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] capabilities {
    /// @brief MPI is available and initialized
    bool has_mpi = false;

    /// @brief MPI thread level actually provided
    init_options::thread_level mpi_thread_level = init_options::thread_level::single;

    /// @brief Number of MPI ranks (1 if no MPI)
    rank_t num_ranks = 1;

    /// @brief This process's rank (0 if no MPI)
    rank_t my_rank = 0;

    /// @brief CUDA is available
    bool has_cuda = false;

    /// @brief Number of CUDA devices visible
    int num_cuda_devices = 0;

    /// @brief Shared memory backend available
    bool has_shared_memory = true;

    /// @brief NCCL is available
    bool has_nccl = false;

    /// @brief Check if running in distributed mode (multiple ranks)
    [[nodiscard]] bool is_distributed() const noexcept {
        return num_ranks > 1;
    }

    /// @brief Check if running with GPU support
    [[nodiscard]] bool has_gpu() const noexcept {
        return has_cuda && num_cuda_devices > 0;
    }
};

// ============================================================================
// Forward Declarations
// ============================================================================

class environment;

// ============================================================================
// Environment Handle
// ============================================================================

/// @brief Global environment handle (singleton-like access)
/// @details Provides access to the initialized DTL environment.
///          Must call dtl::init() before accessing.
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] environment_handle {
public:
    /// @brief Check if environment is initialized
    [[nodiscard]] static bool is_initialized() noexcept;

    /// @brief Get capabilities (must be initialized)
    [[nodiscard]] static const capabilities& caps() noexcept;

    /// @brief Get my rank (must be initialized)
    [[nodiscard]] static rank_t rank() noexcept;

    /// @brief Get number of ranks (must be initialized)
    [[nodiscard]] static rank_t size() noexcept;

private:
    friend class scoped_environment;
    friend result<void> init(int& argc, char**& argv, const init_options& options);
    friend result<void> init(const init_options& options);
    friend result<void> finalize() noexcept;
    friend void at_finalize(std::function<void()> callback);
    static std::unique_ptr<environment> instance_;
};

// ============================================================================
// Scoped Environment
// ============================================================================

/// @brief RAII guard for DTL environment lifetime
/// @details Initializes DTL on construction, finalizes on destruction.
///          Typically used in main().
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
///
/// @par Usage:
/// @code
/// int main(int argc, char** argv) {
///     dtl::scoped_environment env(argc, argv);
///     // or with options:
///     // dtl::scoped_environment env(argc, argv, dtl::init_options{...});
///
///     // DTL is now initialized, use distributed containers, etc.
///
///     return 0;
/// }  // DTL finalized here
/// @endcode
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] scoped_environment {
public:
    /// @brief Initialize with command line arguments
    explicit scoped_environment(int& argc, char**& argv);

    /// @brief Initialize with command line arguments and options
    scoped_environment(int& argc, char**& argv, const init_options& options);

    /// @brief Initialize without command line (for testing/embedded use)
    scoped_environment();

    /// @brief Initialize without command line but with options
    explicit scoped_environment(const init_options& options);

    /// @brief Non-copyable
    scoped_environment(const scoped_environment&) = delete;
    scoped_environment& operator=(const scoped_environment&) = delete;

    /// @brief Non-movable (singleton-like)
    scoped_environment(scoped_environment&&) = delete;
    scoped_environment& operator=(scoped_environment&&) = delete;

    /// @brief Finalize DTL environment
    ~scoped_environment();

    /// @brief Get capabilities
    [[nodiscard]] const capabilities& caps() const noexcept;

    /// @brief Get my rank
    [[nodiscard]] rank_t rank() const noexcept;

    /// @brief Get number of ranks
    [[nodiscard]] rank_t size() const noexcept;

private:
    void initialize(int* argc, char*** argv, const init_options& options);
};

// ============================================================================
// Free Functions
// ============================================================================

/// @brief Initialize DTL environment (free function alternative)
/// @details Alternative to scoped_environment for cases where RAII
///          isn't suitable. Must call dtl::finalize() to clean up.
///
/// @par Usage:
/// @code
/// dtl::init(argc, argv);
/// // ... use DTL ...
/// dtl::finalize();
/// @endcode
result<void> init(int& argc, char**& argv, const init_options& options = {});

/// @brief Initialize without command line
result<void> init(const init_options& options = {});

/// @brief Finalize DTL environment
/// @details Must be called if using dtl::init() (not needed with scoped_environment)
result<void> finalize() noexcept;

/// @brief Check if DTL is initialized
[[nodiscard]] bool is_initialized() noexcept;

/// @brief Get capabilities (must be initialized)
[[nodiscard]] const capabilities& caps() noexcept;

/// @brief Shorthand for my rank
[[nodiscard]] inline rank_t rank() noexcept {
    return environment_handle::rank();
}

/// @brief Shorthand for number of ranks
[[nodiscard]] inline rank_t size() noexcept {
    return environment_handle::size();
}

/// @brief Register a callback to be invoked at finalization
/// @details Callbacks are invoked in reverse order of registration
///          (LIFO) during finalize().
void at_finalize(std::function<void()> callback);

// ============================================================================
// Internal Environment Implementation
// ============================================================================

/// @brief Internal environment state holder
/// @deprecated Use dtl::environment from <dtl/core/environment.hpp>
class [[deprecated("Use dtl::environment from <dtl/core/environment.hpp>")]] environment {
public:
    capabilities caps;
    std::vector<std::function<void()>> finalize_callbacks;
    bool initialized = false;

    environment() = default;

    /// @brief Initialize the environment
    /// @details Single-process fallback initialization. Sets up default
    ///          capabilities (1 rank, no MPI/CUDA/NCCL). Backend-specific
    ///          initialization (MPI_Init, CUDA device setup, etc.) is handled
    ///          by dedicated backend modules (e.g., mpi_environment), not here.
    /// @param argc Pointer to argc (unused in single-process mode)
    /// @param argv Pointer to argv (unused in single-process mode)
    /// @param options Initialization options controlling which backends to enable
    /// @return Success, or invalid_state if already initialized
    result<void> do_init(int* argc, char*** argv, const init_options& options) {
        if (initialized) {
            return status{status_code::invalid_state, no_rank, "DTL already initialized"};
        }

        // Initialize capabilities with single-process defaults
        caps.has_mpi = false;
        caps.mpi_thread_level = options.threading;
        caps.num_ranks = 1;
        caps.my_rank = 0;
        caps.has_cuda = false;
        caps.num_cuda_devices = 0;
        caps.has_shared_memory = options.enable_shared_memory;
        caps.has_nccl = false;

        // Suppress unused parameter warnings (used by backend-specific subclasses)
        (void)argc;
        (void)argv;
        (void)options;

        initialized = true;
        return {};
    }

    /// @brief Finalize the environment
    /// @details Runs registered finalize callbacks in reverse (LIFO) order,
    ///          then marks the environment as uninitialized. Backend-specific
    ///          teardown (MPI_Finalize, CUDA cleanup, etc.) is handled by
    ///          dedicated backend modules, not here.
    /// @return Success, or invalid_state if not initialized
    result<void> do_finalize() {
        if (!initialized) {
            return status{status_code::invalid_state, no_rank, "DTL not initialized"};
        }

        // Run finalize callbacks in reverse order (LIFO)
        for (auto it = finalize_callbacks.rbegin(); it != finalize_callbacks.rend(); ++it) {
            (*it)();
        }
        finalize_callbacks.clear();

        initialized = false;
        return {};
    }
};

// ============================================================================
// Static Storage
// ============================================================================

inline std::unique_ptr<environment> environment_handle::instance_ = nullptr;

inline bool environment_handle::is_initialized() noexcept {
    return instance_ != nullptr && instance_->initialized;
}

inline const capabilities& environment_handle::caps() noexcept {
    DTL_ASSERT(instance_ != nullptr);
    return instance_->caps;
}

inline rank_t environment_handle::rank() noexcept {
    if (instance_ == nullptr) return 0;
    return instance_->caps.my_rank;
}

inline rank_t environment_handle::size() noexcept {
    if (instance_ == nullptr) return 1;
    return instance_->caps.num_ranks;
}

// ============================================================================
// scoped_environment Implementation
// ============================================================================

inline scoped_environment::scoped_environment(int& argc, char**& argv)
    : scoped_environment(argc, argv, init_options{}) {}

inline scoped_environment::scoped_environment(int& argc, char**& argv, const init_options& options) {
    initialize(&argc, &argv, options);
}

inline scoped_environment::scoped_environment()
    : scoped_environment(init_options{}) {}

inline scoped_environment::scoped_environment(const init_options& options) {
    initialize(nullptr, nullptr, options);
}

inline scoped_environment::~scoped_environment() {
    if (environment_handle::is_initialized()) {
        environment_handle::instance_->do_finalize();
        environment_handle::instance_.reset();
    }
}

inline void scoped_environment::initialize(int* argc, char*** argv, const init_options& options) {
    if (environment_handle::instance_ != nullptr) {
        // Already initialized - this is an error
        DTL_ASSERT(false && "DTL already initialized - cannot create multiple scoped_environments");
        return;
    }

    environment_handle::instance_ = std::make_unique<environment>();
    auto result = environment_handle::instance_->do_init(argc, argv, options);
    if (!result) {
        environment_handle::instance_.reset();
        // In production, we might throw or store the error
        DTL_ASSERT(false && "DTL initialization failed");
    }
}

inline const capabilities& scoped_environment::caps() const noexcept {
    return environment_handle::caps();
}

inline rank_t scoped_environment::rank() const noexcept {
    return environment_handle::rank();
}

inline rank_t scoped_environment::size() const noexcept {
    return environment_handle::size();
}

// ============================================================================
// Free Function Implementation
// ============================================================================

inline bool is_initialized() noexcept {
    return environment_handle::is_initialized();
}

inline const capabilities& caps() noexcept {
    return environment_handle::caps();
}

inline result<void> init(int& argc, char**& argv, const init_options& options) {
    if (environment_handle::instance_ != nullptr) {
        return status{status_code::invalid_state, no_rank, "DTL already initialized"};
    }

    environment_handle::instance_ = std::make_unique<environment>();
    auto result = environment_handle::instance_->do_init(&argc, &argv, options);
    if (!result) {
        environment_handle::instance_.reset();
    }
    return result;
}

inline result<void> init(const init_options& options) {
    if (environment_handle::instance_ != nullptr) {
        return status{status_code::invalid_state, no_rank, "DTL already initialized"};
    }

    environment_handle::instance_ = std::make_unique<environment>();
    auto result = environment_handle::instance_->do_init(nullptr, nullptr, options);
    if (!result) {
        environment_handle::instance_.reset();
    }
    return result;
}

inline result<void> finalize() noexcept {
    if (environment_handle::instance_ == nullptr) {
        return status{status_code::invalid_state, no_rank, "DTL not initialized"};
    }

    auto result = environment_handle::instance_->do_finalize();
    environment_handle::instance_.reset();
    return result;
}

inline void at_finalize(std::function<void()> callback) {
    if (environment_handle::instance_ != nullptr && environment_handle::instance_->initialized) {
        environment_handle::instance_->finalize_callbacks.push_back(std::move(callback));
    }
}

}  // namespace legacy
}  // namespace dtl
