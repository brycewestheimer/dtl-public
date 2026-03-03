// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file runtime_registry.hpp
/// @brief Internal Meyer's singleton for backend lifecycle management
/// @details Holds all process-global backend state (MPI, CUDA, HIP, NCCL, SHMEM)
///          that was previously stored as static inline members of environment.
///          The registry uses a function-local static (Meyer's singleton) for
///          well-defined initialization order across DSO boundaries.
///
///          This is an internal implementation detail. Users should interact
///          with dtl::environment, not this class directly.
///
///          The singleton instance and all method bodies live in
///          libdtl_runtime.so. This header is declaration-only.
///
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/environment_options.hpp>
#include <dtl/runtime/detail/runtime_export.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string_view>

namespace dtl::runtime {

/// @brief Process-global singleton managing backend initialization and finalization
/// @details All mutable process-global state lives here. The environment class
///          is a lightweight handle/view that delegates to this registry.
///
///          Thread safety: all public methods are thread-safe (guarded by mtx_).
///          The singleton itself is safe to construct from any thread (C++11
///          function-local static guarantee).
///
///          The singleton and all method implementations reside in
///          libdtl_runtime.so to ensure exactly one copy exists process-wide,
///          even when multiple static libraries include DTL headers.
class runtime_registry {
public:
    /// @brief Get the singleton instance (Meyer's singleton)
    /// @return Reference to the process-global registry
    /// @details The function-local static lives in libdtl_runtime.so,
    ///          ensuring a single instance across all translation units.
    DTL_RUNTIME_API static runtime_registry& instance();

    /// @brief Acquire a reference to the runtime
    /// @param opts Backend configuration options (only used on first acquire)
    /// @param argc Pointer to argc from main() (may be nullptr)
    /// @param argv Pointer to argv from main() (may be nullptr)
    /// @details First acquire (refcount 0 -> 1) initializes backends.
    ///          Subsequent acquires only increment the reference count.
    DTL_RUNTIME_API void acquire(environment_options opts, int* argc, char*** argv);

    /// @brief Release a reference to the runtime
    /// @details Last release (refcount 1 -> 0) finalizes backends in reverse order.
    DTL_RUNTIME_API void release();

    // -------------------------------------------------------------------------
    // State Queries
    // -------------------------------------------------------------------------

    [[nodiscard]] DTL_RUNTIME_API bool is_initialized() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API size_t ref_count() const noexcept;

    [[nodiscard]] DTL_RUNTIME_API bool has_mpi() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API bool has_cuda() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API bool has_hip() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API bool has_nccl() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API bool has_shmem() const noexcept;

    [[nodiscard]] DTL_RUNTIME_API int mpi_thread_level() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API const char* mpi_thread_level_name() const noexcept;

    [[nodiscard]] DTL_RUNTIME_API thread_support_level thread_level() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API std::string_view thread_level_name() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API thread_support_level thread_level_for_backend(
        std::string_view backend) const noexcept;
    [[nodiscard]] DTL_RUNTIME_API determinism_options determinism_options_config() const noexcept;
    [[nodiscard]] DTL_RUNTIME_API bool deterministic_mode_enabled() const noexcept;

    [[nodiscard]] DTL_RUNTIME_API std::shared_ptr<const std::atomic<std::uint64_t>>
    lifetime_generation_token() const;
    [[nodiscard]] DTL_RUNTIME_API std::uint64_t lifetime_generation() const noexcept;

    // Non-copyable, non-movable
    runtime_registry(const runtime_registry&) = delete;
    runtime_registry& operator=(const runtime_registry&) = delete;
    runtime_registry(runtime_registry&&) = delete;
    runtime_registry& operator=(runtime_registry&&) = delete;

private:
    runtime_registry() = default;
    ~runtime_registry() = default;

    // -------------------------------------------------------------------------
    // State (must remain in header for class layout)
    // -------------------------------------------------------------------------

    mutable std::mutex mtx_;
    std::atomic<size_t> ref_count_{0};
    std::unique_ptr<environment_options> opts_;
    int* argc_{nullptr};
    char*** argv_{nullptr};
    int dummy_argc_{0};
    char** dummy_argv_{nullptr};

    bool mpi_available_{false};
    bool cuda_available_{false};
    bool hip_available_{false};
    bool nccl_available_{false};
    bool shmem_available_{false};
    int  mpi_thread_level_{-1};
    bool mpi_initialized_by_dtl_{false};
    determinism_options determinism_opts_{};
    std::shared_ptr<std::atomic<std::uint64_t>> lifetime_generation_{};

    // -------------------------------------------------------------------------
    // Backend Initialization / Finalization (defined in runtime_registry.cpp)
    // -------------------------------------------------------------------------

    void initialize_backends();
    void initialize_mpi();
    void check_thread_level(int requested_level);
    void initialize_cuda();
    void initialize_hip();
    void initialize_nccl();
    void initialize_shmem();

    void finalize_backends();
    void finalize_mpi();
    void finalize_cuda();
    void finalize_hip();
    void finalize_nccl();
    void finalize_shmem();
};

}  // namespace dtl::runtime
