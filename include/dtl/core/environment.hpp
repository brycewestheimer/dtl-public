// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file environment.hpp
/// @brief RAII handle/view for DTL backend lifecycle management
/// @details Each environment instance is a lightweight handle that references
///          a process-global runtime_registry (Meyer's singleton). The registry
///          manages backend init/finalize (MPI, CUDA, SHMEM, NCCL, HIP) with
///          reference counting. Each environment owns a per-instance MPI
///          communicator (dup'd from MPI_COMM_WORLD or injected via from_comm)
///          providing communicator isolation for multi-library composition.
///
///          The first environment construction triggers backend initialization;
///          the last destruction triggers finalization in reverse order.
///
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/environment_options.hpp>
#include <dtl/core/context.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/handle/handle.hpp>
#include <dtl/runtime/runtime_registry.hpp>
#include <dtl/error/result.hpp>
#include <backends/mpi/mpi_lifecycle.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <string>
#include <string_view>
#include <utility>

namespace dtl {

// =============================================================================
// Environment Handle/View
// =============================================================================

/// @brief RAII handle for backend lifecycle management with per-instance
///        communicator isolation
/// @details Each environment instance:
///          - Holds a reference to the process-global runtime_registry
///          - Owns a per-instance MPI communicator (dup'd from MPI_COMM_WORLD
///            or from an injected communicator)
///          - Provides instance-based backend queries that delegate to the registry
///
///          Multiple environment instances can coexist independently, each with
///          its own communicator. This enables safe multi-library composition
///          where each library creates its own environment without cross-talk.
///
///          The class is move-only: moving transfers communicator ownership.
///          Copying is deleted because each instance owns a unique dup'd comm.
///
/// @note The legacy `environment_base` class from utility/environment.hpp
///       is now in `dtl::legacy::` namespace and is deprecated.
/// @since 0.1.0
class environment {
public:
    // -------------------------------------------------------------------------
    // Construction / Destruction
    // -------------------------------------------------------------------------

    /// @brief Construct an environment handle with argc/argv for MPI
    /// @param argc Reference to argc from main() - passed to MPI_Init_thread
    /// @param argv Reference to argv from main() - passed to MPI_Init_thread
    /// @param opts Backend configuration options (only used by the first construction)
    /// @details First construction initializes backends per options.
    ///          Subsequent constructions increment the reference count only.
    ///          Each environment gets its own MPI communicator (dup'd from
    ///          MPI_COMM_WORLD) for isolation.
    ///          This overload is preferred for multi-rank MPI programs.
    environment(int& argc, char**& argv,
                environment_options opts = environment_options::defaults())
        : domain_(opts.domain)
    {
        runtime::runtime_registry::instance().acquire(std::move(opts), &argc, &argv);
        dup_world_comm();
    }

    /// @brief Construct an environment handle without argc/argv
    /// @param opts Backend configuration options (only used by the first construction)
    /// @details First construction initializes backends per options.
    ///          Subsequent constructions increment the reference count only.
    /// @note Some MPI implementations may not work correctly with this overload.
    ///       For multi-rank MPI programs, prefer the (argc, argv) overload.
    explicit environment(environment_options opts = environment_options::defaults())
        : domain_(opts.domain)
    {
        runtime::runtime_registry::instance().acquire(std::move(opts), nullptr, nullptr);
        dup_world_comm();
    }

    /// @brief Destroy an environment handle
    /// @details Frees the per-instance communicator (if owned), then releases
    ///          the registry reference. When the last handle is destroyed,
    ///          backends are finalized in reverse order.
    ~environment() {
        free_comm();
        runtime::runtime_registry::instance().release();
    }

    // Non-copyable (each instance owns a unique dup'd communicator)
    environment(const environment&) = delete;
    environment& operator=(const environment&) = delete;

    // Movable (transfers communicator ownership)
    environment(environment&& other) noexcept
        : domain_(std::move(other.domain_))
#if DTL_ENABLE_MPI
        , comm_(other.comm_)
        , owns_comm_(other.owns_comm_)
#endif
    {
#if DTL_ENABLE_MPI
        other.comm_ = MPI_COMM_NULL;
        other.owns_comm_ = false;
#endif
        // Acquire a new registry reference for this handle (the moved-from
        // handle will still release its reference in its destructor)
        runtime::runtime_registry::instance().acquire(
            environment_options::minimal(), nullptr, nullptr);
    }

    environment& operator=(environment&& other) noexcept {
        if (this != &other) {
            // Release our current communicator
            free_comm();

            domain_ = std::move(other.domain_);
#if DTL_ENABLE_MPI
            comm_ = other.comm_;
            owns_comm_ = other.owns_comm_;
            other.comm_ = MPI_COMM_NULL;
            other.owns_comm_ = false;
#endif
            // Note: both handles already hold registry references;
            // no acquire/release needed for the swap
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Factory: Communicator Injection (NEW in 1.4.0)
    // -------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Create an environment from an existing MPI communicator
    /// @param comm MPI communicator to use (will be dup'd for isolation)
    /// @param opts Backend configuration options
    /// @return New environment with a dup'd copy of the provided communicator
    /// @details This is the primary API for library authors: pass in the
    ///          communicator your library was given, and DTL will create an
    ///          isolated copy. Multiple libraries can each call from_comm()
    ///          with the same or different communicators without interference.
    /// @since 0.1.0
    [[nodiscard]] static environment from_comm(
        MPI_Comm comm,
        environment_options opts = environment_options::defaults())
    {
        return environment(comm, std::move(opts));
    }
#endif

    // -------------------------------------------------------------------------
    // Instance Queries (delegate to registry)
    // -------------------------------------------------------------------------

    /// @brief Get the named domain label for this environment
    /// @return Domain name (for diagnostics; defaults to "default")
    /// @since 0.1.0
    [[nodiscard]] std::string_view domain() const noexcept {
        return domain_;
    }

    /// @brief Check if MPI backend is available and was initialized
    /// @return true if MPI was successfully initialized or adopted
    [[nodiscard]] bool has_mpi() const noexcept {
        return runtime::runtime_registry::instance().has_mpi();
    }

    /// @brief Check if CUDA backend is available and was initialized
    /// @return true if CUDA was successfully initialized
    [[nodiscard]] bool has_cuda() const noexcept {
        return runtime::runtime_registry::instance().has_cuda();
    }

    /// @brief Check if HIP/ROCm backend is available and was initialized
    /// @return true if HIP was successfully initialized
    [[nodiscard]] bool has_hip() const noexcept {
        return runtime::runtime_registry::instance().has_hip();
    }

    /// @brief Check if NCCL is available (compile-time detection)
    /// @return true if DTL was built with NCCL support and it was not disabled
    [[nodiscard]] bool has_nccl() const noexcept {
        return runtime::runtime_registry::instance().has_nccl();
    }

    /// @brief Check if SHMEM (OpenSHMEM) backend is available and was initialized
    /// @return true if SHMEM was successfully initialized
    [[nodiscard]] bool has_shmem() const noexcept {
        return runtime::runtime_registry::instance().has_shmem();
    }

    /// @brief Get the MPI thread support level that was provided
    /// @return Thread level (0-3), or -1 if MPI is not available
    [[nodiscard]] int mpi_thread_level() const noexcept {
        return runtime::runtime_registry::instance().mpi_thread_level();
    }

    /// @brief Get string name for the MPI thread support level
    /// @return Human-readable thread level name (MPI-specific format)
    [[nodiscard]] const char* mpi_thread_level_name() const noexcept {
        return runtime::runtime_registry::instance().mpi_thread_level_name();
    }

    /// @brief Get thread support level as generic enum
    /// @return Thread support level, or single if no backend is available
    /// @since 0.1.0
    [[nodiscard]] thread_support_level thread_level() const noexcept {
        return runtime::runtime_registry::instance().thread_level();
    }

    /// @brief Get thread support level name (generic, not MPI-specific)
    /// @return Human-readable thread level name ("single", "funneled", etc.)
    /// @since 0.1.0
    [[nodiscard]] std::string_view thread_level_name() const noexcept {
        return runtime::runtime_registry::instance().thread_level_name();
    }

    /// @brief Get thread support level for a specific backend
    /// @param backend Backend name ("mpi", "cuda", "shmem", "nccl", "hip")
    /// @return Thread support level for that backend
    /// @since 0.1.0
    [[nodiscard]] thread_support_level thread_level_for_backend(
        std::string_view backend) const noexcept {
        return runtime::runtime_registry::instance().thread_level_for_backend(backend);
    }

    /// @brief Get the per-instance MPI communicator
    /// @return MPI_Comm for this environment (dup'd, isolated)
    /// @note Returns MPI_COMM_NULL if MPI is not available
    /// @since 0.1.0
#if DTL_ENABLE_MPI
    [[nodiscard]] MPI_Comm communicator() const noexcept {
        return comm_;
    }
#endif

    // -------------------------------------------------------------------------
    // Deprecated Static Queries (backward compatibility)
    // -------------------------------------------------------------------------

    /// @brief Check if the environment is currently initialized (refcount > 0)
    /// @return true if at least one environment handle exists
    /// @deprecated Use instance method or check registry directly
    [[deprecated("use instance method")]]
    [[nodiscard]] static bool is_initialized() noexcept {
        return runtime::runtime_registry::instance().is_initialized();
    }

    /// @brief Get the current reference count
    /// @return Number of active environment handles
    /// @deprecated Use instance method or check registry directly
    [[deprecated("use instance method")]]
    [[nodiscard]] static size_t ref_count() noexcept {
        return runtime::runtime_registry::instance().ref_count();
    }

    // -------------------------------------------------------------------------
    // Context Factory Methods (V1.3.0)
    // -------------------------------------------------------------------------

    /// @brief Create a world context with MPI and CPU domains
    /// @return context<mpi_domain, cpu_domain> for the world communicator
    /// @details Creates a context that spans all MPI ranks. Uses the
    ///          per-instance communicator for isolation.
    /// @pre MPI must be available (has_mpi() == true)
    /// @since 0.1.0
    [[nodiscard]] mpi_context make_world_context() const {
        auto& reg = runtime::runtime_registry::instance();
        if (!reg.has_mpi()) {
            return mpi_context{mpi_domain{nullptr}, cpu_domain{}};
        }

#if DTL_ENABLE_MPI
        if (comm_ != MPI_COMM_NULL) {
            auto owned_comm = std::make_shared<mpi::mpi_communicator>(comm_, false);
            auto adapter = std::make_shared<mpi::mpi_comm_adapter>(owned_comm);
            return mpi_context{mpi_domain{std::move(adapter)}, cpu_domain{}};
        }
#endif

        return mpi_context{mpi_domain{}, cpu_domain{}};
    }

    /// @brief Create a world context with MPI, CPU, and CUDA domains
    /// @param device_id CUDA device ID to use for this rank
    /// @return context<mpi_domain, cpu_domain, cuda_domain>
    /// @details Creates a GPU-enabled context for heterogeneous computing.
    /// @pre MPI must be available (has_mpi() == true)
    /// @pre CUDA must be available (has_cuda() == true)
    /// @since 0.1.0
    [[nodiscard]] mpi_cuda_context make_world_context(int device_id) const {
        auto& reg = runtime::runtime_registry::instance();
        if (!reg.has_mpi() || !reg.has_cuda()) {
            return mpi_cuda_context{mpi_domain{nullptr}, cpu_domain{}, cuda_domain{device_id}};
        }

#if DTL_ENABLE_MPI
        if (comm_ != MPI_COMM_NULL) {
            auto owned_comm = std::make_shared<mpi::mpi_communicator>(comm_, false);
            auto adapter = std::make_shared<mpi::mpi_comm_adapter>(owned_comm);
            return mpi_cuda_context{mpi_domain{std::move(adapter)}, cpu_domain{}, cuda_domain{device_id}};
        }
#endif

        return mpi_cuda_context{mpi_domain{}, cpu_domain{}, cuda_domain{device_id}};
    }

    [[nodiscard]] handle::runtime_handle runtime_handle() const {
        return handle::runtime_handle::current();
    }

    [[nodiscard]] handle::comm_handle communicator_handle() const {
        auto runtime = runtime_handle();

#if DTL_ENABLE_MPI
        if (has_mpi() && comm_ != MPI_COMM_NULL) {
            int rank = 0;
            int size = 1;
            (void)MPI_Comm_rank(comm_, &rank);
            (void)MPI_Comm_size(comm_, &size);

            MPI_Comm captured_comm = comm_;
            auto barrier_fn = [captured_comm]() -> result<void> {
                if (!mpi::is_initialized() || mpi::is_finalized() || captured_comm == MPI_COMM_NULL) {
                    return result<void>::failure(
                        status{status_code::invalid_state, no_rank,
                               "MPI backend is not operational for communicator barrier"});
                }

                int err = MPI_Barrier(captured_comm);
                if (err != MPI_SUCCESS) {
                    return result<void>::failure(
                        status{status_code::barrier_failed, no_rank,
                               "MPI_Barrier failed on environment communicator"});
                }

                return result<void>::success();
            };

            return handle::comm_handle{static_cast<rank_t>(rank),
                                       static_cast<rank_t>(size),
                                       std::move(barrier_fn),
                                       std::move(runtime)};
        }
#endif

        return handle::comm_handle::local(std::move(runtime));
    }

    [[nodiscard]] handle::context_handle handle() const {
        auto runtime = runtime_handle();
        auto determinism = runtime::runtime_registry::instance().determinism_options_config();
        return handle::context_handle{communicator_handle(), std::move(runtime), determinism};
    }

    /// @brief Create a CPU-only context (single-process, no MPI)
    /// @return context<cpu_domain>
    /// @details Useful for local testing or non-distributed code.
    /// @since 0.1.0
    [[nodiscard]] cpu_context make_cpu_context() const {
        return cpu_context{cpu_domain{}};
    }

private:
    // -------------------------------------------------------------------------
    // Per-Instance State
    // -------------------------------------------------------------------------

    std::string domain_{"default"};  ///< Named domain for diagnostics (Rule 7)

#if DTL_ENABLE_MPI
    MPI_Comm comm_ = MPI_COMM_NULL;  ///< Per-instance communicator (dup'd)
    bool owns_comm_ = false;          ///< True if we dup'd the communicator
#endif

    // -------------------------------------------------------------------------
    // Private Constructor for from_comm()
    // -------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Private constructor for from_comm() factory
    environment(MPI_Comm user_comm, environment_options opts)
        : domain_(opts.domain)
    {
        runtime::runtime_registry::instance().acquire(std::move(opts), nullptr, nullptr);

        // Dup the user-provided communicator for isolation
        if (runtime::runtime_registry::instance().has_mpi() &&
            user_comm != MPI_COMM_NULL) {
            int err = MPI_Comm_dup(user_comm, &comm_);
            if (err == MPI_SUCCESS) {
                owns_comm_ = true;
            }
        }
    }
#endif

    // -------------------------------------------------------------------------
    // Communicator Helpers
    // -------------------------------------------------------------------------

    void dup_world_comm() {
#if DTL_ENABLE_MPI
        if (runtime::runtime_registry::instance().has_mpi()) {
            int err = MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
            if (err == MPI_SUCCESS) {
                owns_comm_ = true;
            }
        }
#endif
    }

    void free_comm() {
#if DTL_ENABLE_MPI
        if (owns_comm_ && comm_ != MPI_COMM_NULL) {
            // Only free if MPI is still initialized (not finalized)
            if (mpi::is_initialized() && !mpi::is_finalized()) {
                MPI_Comm_free(&comm_);
            }
            comm_ = MPI_COMM_NULL;
            owns_comm_ = false;
        }
#endif
    }
};

/// @brief Backward-compatibility alias for dtl::environment
/// @deprecated Use dtl::environment directly
using environment_guard = environment;

// =============================================================================
// Free Function Context Factories
// =============================================================================

/// @brief Create a world context from an environment
/// @param env The environment to create context from
/// @return context<mpi_domain, cpu_domain> for the world communicator
/// @since 0.1.0
[[nodiscard]] inline mpi_context make_world_context(const environment& env) {
    return env.make_world_context();
}

/// @brief Create a world MPI context using a process-local static environment
/// @return context<mpi_domain, cpu_domain> for the world communicator
/// @details Compatibility helper for tests and existing call sites that expect
///          a no-argument MPI context factory.
[[nodiscard]] inline mpi_context make_mpi_context() {
    static environment env;
    return env.make_world_context();
}

/// @brief Create a GPU-enabled world context from an environment
/// @param env The environment to create context from
/// @param device_id CUDA device ID to use for this rank
/// @return context<mpi_domain, cpu_domain, cuda_domain>
/// @since 0.1.0
[[nodiscard]] inline mpi_cuda_context make_world_context(const environment& env, int device_id) {
    return env.make_world_context(device_id);
}

}  // namespace dtl
