// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file runtime_registry.cpp
/// @brief Implementation of the process-global runtime registry (DSO singleton)
/// @details All method bodies for runtime_registry live here so that the
///          Meyer's singleton (function-local static) resides in exactly one
///          shared object (libdtl_runtime.so). This prevents duplicate
///          singletons when multiple static libraries include DTL headers.
///
/// @since 0.1.0

#include <dtl/runtime/runtime_registry.hpp>
#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/environment_options.hpp>
#include <dtl/error/result.hpp>
#include <backends/mpi/mpi_lifecycle.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

#include <cstdio>
#include <memory>

namespace dtl::runtime {

// =============================================================================
// Singleton
// =============================================================================

runtime_registry& runtime_registry::instance() {
    static runtime_registry r;
    return r;
}

// =============================================================================
// Acquire / Release
// =============================================================================

void runtime_registry::acquire(environment_options opts, int* argc, char*** argv) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (ref_count_.load(std::memory_order_relaxed) == 0) {
        if (!lifetime_generation_) {
            lifetime_generation_ = std::make_shared<std::atomic<std::uint64_t>>(1);
        }
        opts_ = std::make_unique<environment_options>(std::move(opts));
        determinism_opts_ = opts_->determinism;
        if (argc && argv) {
            argc_ = argc;
            argv_ = argv;
        } else {
            dummy_argc_ = 0;
            dummy_argv_ = nullptr;
            argc_ = &dummy_argc_;
            argv_ = &dummy_argv_;
        }
        initialize_backends();
    }

    ref_count_.fetch_add(1, std::memory_order_relaxed);
}

void runtime_registry::release() {
    std::lock_guard<std::mutex> lock(mtx_);

    auto prev = ref_count_.fetch_sub(1, std::memory_order_relaxed);
    if (prev == 1) {
        finalize_backends();
        opts_.reset();
        determinism_opts_ = determinism_options{};
    }
}

// =============================================================================
// State Queries
// =============================================================================

bool runtime_registry::is_initialized() const noexcept {
    return ref_count_.load(std::memory_order_acquire) > 0;
}

size_t runtime_registry::ref_count() const noexcept {
    return ref_count_.load(std::memory_order_acquire);
}

bool runtime_registry::has_mpi() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return mpi_available_;
}

bool runtime_registry::has_cuda() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return cuda_available_;
}

bool runtime_registry::has_hip() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return hip_available_;
}

bool runtime_registry::has_nccl() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return nccl_available_;
}

bool runtime_registry::has_shmem() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return shmem_available_;
}

int runtime_registry::mpi_thread_level() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return mpi_thread_level_;
}

const char* runtime_registry::mpi_thread_level_name() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
#if DTL_ENABLE_MPI
    switch (mpi_thread_level_) {
        case MPI_THREAD_SINGLE:     return "MPI_THREAD_SINGLE";
        case MPI_THREAD_FUNNELED:   return "MPI_THREAD_FUNNELED";
        case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
        case MPI_THREAD_MULTIPLE:   return "MPI_THREAD_MULTIPLE";
        default:                    return "unknown";
    }
#else
    return "MPI_NOT_ENABLED";
#endif
}

thread_support_level runtime_registry::thread_level() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    if (mpi_thread_level_ < 0) {
        return thread_support_level::single;
    }
    return from_mpi_thread_level(mpi_thread_level_);
}

std::string_view runtime_registry::thread_level_name() const noexcept {
    return to_string(thread_level());
}

thread_support_level runtime_registry::thread_level_for_backend(
    std::string_view backend) const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    if (backend == "mpi") {
        if (!mpi_available_ || mpi_thread_level_ < 0) {
            return thread_support_level::single;
        }
        return from_mpi_thread_level(mpi_thread_level_);
    }
    if (backend == "cuda" && cuda_available_) {
        return thread_support_level::multiple;
    }
    if (backend == "hip" && hip_available_) {
        return thread_support_level::multiple;
    }
    if (backend == "nccl" && nccl_available_) {
        return thread_support_level::multiple;
    }
    if (backend == "shmem" && shmem_available_) {
        return thread_support_level::single;
    }
    return thread_support_level::single;
}

determinism_options runtime_registry::determinism_options_config() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return determinism_opts_;
}

bool runtime_registry::deterministic_mode_enabled() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return determinism_opts_.mode == determinism_mode::deterministic;
}

std::shared_ptr<const std::atomic<std::uint64_t>>
runtime_registry::lifetime_generation_token() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return lifetime_generation_;
}

std::uint64_t runtime_registry::lifetime_generation() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!lifetime_generation_) {
        return 0;
    }
    return lifetime_generation_->load(std::memory_order_acquire);
}

// =============================================================================
// Backend Initialization (called under lock, first acquire only)
// =============================================================================

void runtime_registry::initialize_backends() {
    initialize_mpi();
    initialize_cuda();
    initialize_hip();
    initialize_nccl();
    initialize_shmem();
}

void runtime_registry::initialize_mpi() {
    const auto& mpi_opts = opts_->mpi;
    mpi_initialized_by_dtl_ = false;

    switch (mpi_opts.ownership) {
        case backend_ownership::dtl_owns: {
            const bool was_initialized = mpi::is_initialized();
            auto res = mpi::initialize(argc_, argv_,
                                       mpi_opts.thread_level,
                                       mpi_opts.allow_thread_fallback);
            if (res.has_value()) {
                mpi_available_ = true;
                mpi_thread_level_ = res.value();
                mpi_initialized_by_dtl_ = !was_initialized;
            }
            break;
        }
        case backend_ownership::adopt_external: {
            auto res = mpi::verify_initialized();
            if (res.has_value()) {
                mpi_available_ = true;
                auto level_res = mpi::query_thread_level();
                mpi_thread_level_ = level_res.has_value() ? level_res.value() : -1;
                mpi_initialized_by_dtl_ = false;
            }
            break;
        }
        case backend_ownership::optional: {
            if (!mpi::is_initialized() && !mpi::is_finalized()) {
                const bool was_initialized = mpi::is_initialized();
                auto res = mpi::initialize(argc_, argv_,
                                           mpi_opts.thread_level,
                                           mpi_opts.allow_thread_fallback);
                if (res.has_value()) {
                    mpi_available_ = true;
                    mpi_thread_level_ = res.value();
                    mpi_initialized_by_dtl_ = !was_initialized;
                }
            } else if (mpi::is_initialized()) {
                mpi_available_ = true;
                auto level_res = mpi::query_thread_level();
                mpi_thread_level_ = level_res.has_value() ? level_res.value() : -1;
                mpi_initialized_by_dtl_ = false;
            }
            break;
        }
        case backend_ownership::disabled:
            break;
    }

    if (mpi_available_) {
        check_thread_level(mpi_opts.thread_level);
    }
}

void runtime_registry::check_thread_level([[maybe_unused]] int requested_level) {
    if (mpi_thread_level_ < 0) return;

#if DTL_ENABLE_MPI
    if (mpi_thread_level_ < requested_level) {
        auto level_name = [](int level) -> const char* {
            switch (level) {
                case MPI_THREAD_SINGLE:     return "MPI_THREAD_SINGLE";
                case MPI_THREAD_FUNNELED:   return "MPI_THREAD_FUNNELED";
                case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
                case MPI_THREAD_MULTIPLE:   return "MPI_THREAD_MULTIPLE";
                default:                    return "unknown";
            }
        };
        std::fprintf(stderr,
            "[DTL WARNING] MPI thread level is %s (%d), but %s (%d) was requested. "
            "Some async operations may require MPI_THREAD_MULTIPLE (%d).\n",
            level_name(mpi_thread_level_), mpi_thread_level_,
            level_name(requested_level), requested_level,
            MPI_THREAD_MULTIPLE);
    }
#endif
}

void runtime_registry::initialize_cuda() {
    const auto& cuda_opts = opts_->cuda;

    switch (cuda_opts.ownership) {
        case backend_ownership::dtl_owns:
        case backend_ownership::optional: {
#if DTL_ENABLE_CUDA
            int device_count = 0;
            cudaError_t err = cudaGetDeviceCount(&device_count);
            if (err == cudaSuccess && device_count > 0) {
                int device_id = cuda_opts.device_id;
                if (device_id < 0) {
                    device_id = 0;
                }
                if (device_id < device_count) {
                    err = cudaSetDevice(device_id);
                    if (err == cudaSuccess) {
                        if (cuda_opts.eager_context) {
                            cudaFree(nullptr);
                        }
                        cuda_available_ = true;
                    }
                }
            }
#endif
            break;
        }
        case backend_ownership::adopt_external: {
#if DTL_ENABLE_CUDA
            int current_device = -1;
            cudaError_t err = cudaGetDevice(&current_device);
            if (err == cudaSuccess && current_device >= 0) {
                cuda_available_ = true;
            }
#endif
            break;
        }
        case backend_ownership::disabled:
            break;
    }
}

void runtime_registry::initialize_hip() {
    const auto& hip_opts = opts_->hip;

    switch (hip_opts.ownership) {
        case backend_ownership::dtl_owns:
        case backend_ownership::optional: {
#if DTL_ENABLE_HIP
            int device_count = 0;
            hipError_t err = hipGetDeviceCount(&device_count);
            if (err == hipSuccess && device_count > 0) {
                int device_id = hip_opts.device_id;
                if (device_id < 0) {
                    device_id = 0;
                }
                if (device_id < device_count) {
                    err = hipSetDevice(device_id);
                    if (err == hipSuccess) {
                        if (hip_opts.eager_context) {
                            hipFree(nullptr);
                        }
                        hip_available_ = true;
                    }
                }
            }
#endif
            break;
        }
        case backend_ownership::adopt_external: {
#if DTL_ENABLE_HIP
            int current_device = -1;
            hipError_t err = hipGetDevice(&current_device);
            if (err == hipSuccess && current_device >= 0) {
                hip_available_ = true;
            }
#endif
            break;
        }
        case backend_ownership::disabled:
            break;
    }
}

void runtime_registry::initialize_nccl() {
    const auto& nccl_opts = opts_->nccl;

    switch (nccl_opts.ownership) {
        case backend_ownership::dtl_owns:
        case backend_ownership::adopt_external:
        case backend_ownership::optional: {
#if DTL_ENABLE_NCCL
            nccl_available_ = true;
#endif
            break;
        }
        case backend_ownership::disabled:
            break;
    }
}

void runtime_registry::initialize_shmem() {
    const auto& shmem_opts = opts_->shmem;

    switch (shmem_opts.ownership) {
        case backend_ownership::dtl_owns: {
#if DTL_ENABLE_SHMEM
            if (shmem_opts.heap_size > 0) {
                // Advisory for implementations that support it
            }
            shmem_init();
            shmem_available_ = true;
#endif
            break;
        }
        case backend_ownership::optional: {
#if DTL_ENABLE_SHMEM
            shmem_init();
            shmem_available_ = true;
#endif
            break;
        }
        case backend_ownership::adopt_external: {
#if DTL_ENABLE_SHMEM
            shmem_available_ = true;
#endif
            break;
        }
        case backend_ownership::disabled:
            break;
    }
}

// =============================================================================
// Backend Finalization (called under lock, last release only)
// =============================================================================

void runtime_registry::finalize_backends() {
    finalize_shmem();
    finalize_nccl();
    finalize_hip();
    finalize_cuda();
    finalize_mpi();

    mpi_available_ = false;
    cuda_available_ = false;
    hip_available_ = false;
    nccl_available_ = false;
    shmem_available_ = false;
    mpi_thread_level_ = -1;
    mpi_initialized_by_dtl_ = false;

    if (lifetime_generation_) {
        lifetime_generation_->fetch_add(1, std::memory_order_acq_rel);
    }
}

void runtime_registry::finalize_mpi() {
    if (!mpi_available_ || !mpi_initialized_by_dtl_) {
        return;
    }
    (void)mpi::finalize_mpi();
}

void runtime_registry::finalize_cuda() {
    if (!cuda_available_) return;
    // DTL does NOT call cudaDeviceReset()
}

void runtime_registry::finalize_hip() {
    if (!hip_available_) return;
    // Same policy as CUDA: DTL does not call hipDeviceReset()
}

void runtime_registry::finalize_nccl() {
    // NCCL communicators are destroyed individually, not globally
}

void runtime_registry::finalize_shmem() {
    if (!shmem_available_) return;

    if (opts_ && opts_->shmem.ownership == backend_ownership::dtl_owns) {
#if DTL_ENABLE_SHMEM
        shmem_finalize();
#endif
    }
}

}  // namespace dtl::runtime
