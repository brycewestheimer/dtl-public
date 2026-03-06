// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_environment.cpp
 * @brief DTL C bindings - Environment implementation
 * @since 0.1.0, updated 1.4.0 (instance-based queries via registry)
 */

#include <dtl/bindings/c/dtl_environment.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>

#include "dtl_internal.hpp"

#include <dtl/core/environment.hpp>
#include <dtl/runtime/runtime_registry.hpp>

#include <cstring>

#ifndef DTL_C_EXPORTS
#define DTL_C_EXPORTS
#endif

// ============================================================================
// Environment Lifecycle
// ============================================================================

extern "C" {

dtl_status dtl_environment_create(dtl_environment_t* env) {
    if (!env) {
        return DTL_ERROR_NULL_POINTER;
    }

    // Allocate the handle struct
    dtl_environment_s* handle = nullptr;
    try {
        handle = new dtl_environment_s();
        handle->impl = new dtl::environment();
        handle->magic = dtl_environment_s::VALID_MAGIC;
    } catch (const std::exception&) {
        if (handle) {
            delete handle->impl;
        }
        delete handle;
        return DTL_ERROR_ALLOCATION_FAILED;
    } catch (...) {
        if (handle) {
            delete handle->impl;
        }
        delete handle;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    *env = handle;
    return DTL_SUCCESS;
}

dtl_status dtl_environment_create_with_args(dtl_environment_t* env,
                                             int* argc, char*** argv) {
    if (!env) {
        return DTL_ERROR_NULL_POINTER;
    }

    // Allocate the handle struct
    dtl_environment_s* handle = nullptr;
    try {
        handle = new dtl_environment_s();
        if (argc && argv) {
            handle->impl = new dtl::environment(*argc, *argv);
        } else {
            // Fall back to no-args constructor if argc/argv not provided
            handle->impl = new dtl::environment();
        }
        handle->magic = dtl_environment_s::VALID_MAGIC;
    } catch (const std::exception&) {
        if (handle) {
            delete handle->impl;
        }
        delete handle;
        return DTL_ERROR_ALLOCATION_FAILED;
    } catch (...) {
        if (handle) {
            delete handle->impl;
        }
        delete handle;
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    *env = handle;
    return DTL_SUCCESS;
}

void dtl_environment_destroy(dtl_environment_t env) {
    if (!env || env->magic != dtl_environment_s::VALID_MAGIC) {
        return;
    }

    // Invalidate magic before deletion to prevent use-after-free
    env->magic = 0;

    // Destroy the C++ environment object, which decrements the internal
    // reference count. If this was the last handle, backends will be finalized.
    delete env->impl;
    env->impl = nullptr;

    delete env;
}

// ============================================================================
// Environment State Queries
// ============================================================================

int dtl_environment_is_initialized(void) {
    return dtl::runtime::runtime_registry::instance().is_initialized() ? 1 : 0;
}

dtl_size_t dtl_environment_ref_count(void) {
    return static_cast<dtl_size_t>(
        dtl::runtime::runtime_registry::instance().ref_count());
}

const char* dtl_environment_domain(dtl_environment_t env) {
    if (!is_valid_environment(env) || !env->impl) {
        return "unknown";
    }

    // Store the domain string in a thread-local buffer for C compatibility
    // (the string_view from domain() is backed by the environment's std::string)
    static thread_local std::string domain_buf;
    domain_buf = std::string(env->impl->domain());
    return domain_buf.c_str();
}

// ============================================================================
// Backend Availability (delegate to registry)
// ============================================================================

int dtl_environment_has_mpi(void) {
    return dtl::runtime::runtime_registry::instance().has_mpi() ? 1 : 0;
}

int dtl_environment_has_cuda(void) {
    return dtl::runtime::runtime_registry::instance().has_cuda() ? 1 : 0;
}

int dtl_environment_has_hip(void) {
    return dtl::runtime::runtime_registry::instance().has_hip() ? 1 : 0;
}

int dtl_environment_has_nccl(void) {
    return dtl::runtime::runtime_registry::instance().has_nccl() ? 1 : 0;
}

int dtl_environment_has_shmem(void) {
    return dtl::runtime::runtime_registry::instance().has_shmem() ? 1 : 0;
}

int dtl_environment_mpi_thread_level(void) {
    return dtl::runtime::runtime_registry::instance().mpi_thread_level();
}

// ============================================================================
// Context Factory Methods
// ============================================================================

dtl_status dtl_environment_make_world_context(dtl_environment_t env,
                                               dtl_context_t* ctx) {
    if (!ctx) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!is_valid_environment(env)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Create C++ context from environment
    auto cpp_ctx = env->impl->make_world_context();

    // Allocate C context handle
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Populate from the C++ context
    impl->rank = cpp_ctx.rank();
    impl->size = cpp_ctx.size();
    impl->device_id = -1;  // CPU-only world context
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = dtl_context_s::HAS_CPU;
    impl->error_handler = nullptr;
    impl->error_handler_user_data = nullptr;

#ifdef DTL_HAS_MPI
    if (env->impl->has_mpi()) {
        MPI_Comm source_comm = env->impl->communicator();
        if (source_comm == MPI_COMM_NULL) {
            delete impl;
            return DTL_ERROR_MPI;
        }

        int err = MPI_Comm_dup(source_comm, &impl->comm);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
        impl->owns_comm = true;
        impl->initialized_mpi = false;
        impl->finalize_mpi = false;
        impl->domain_flags |= dtl_context_s::HAS_MPI;
    }
#endif

    *ctx = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_environment_make_world_context_gpu(dtl_environment_t env,
                                                   int device_id,
                                                   dtl_context_t* ctx) {
    if (!ctx) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!is_valid_environment(env)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    if (!env->impl->has_cuda()) {
        return DTL_ERROR_BACKEND_UNAVAILABLE;
    }

    // Create C++ GPU context from environment
    auto cpp_ctx = env->impl->make_world_context(device_id);

    // Allocate C context handle
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // Populate from the C++ context
    impl->rank = cpp_ctx.rank();
    impl->size = cpp_ctx.size();
    impl->device_id = device_id;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = dtl_context_s::HAS_CPU;
    impl->error_handler = nullptr;
    impl->error_handler_user_data = nullptr;

    // Mark CUDA domain if available
    if (env->impl->has_cuda()) {
        impl->domain_flags |= dtl_context_s::HAS_CUDA;
    }

#ifdef DTL_HAS_MPI
    if (env->impl->has_mpi()) {
        MPI_Comm source_comm = env->impl->communicator();
        if (source_comm == MPI_COMM_NULL) {
            delete impl;
            return DTL_ERROR_MPI;
        }

        int err = MPI_Comm_dup(source_comm, &impl->comm);
        if (err != MPI_SUCCESS) {
            delete impl;
            return DTL_ERROR_MPI;
        }
        (void)MPI_Comm_set_errhandler(impl->comm, MPI_ERRORS_RETURN);
        impl->owns_comm = true;
        impl->initialized_mpi = false;
        impl->finalize_mpi = false;
        impl->domain_flags |= dtl_context_s::HAS_MPI;
    }
#endif

    *ctx = impl;
    return DTL_SUCCESS;
}

dtl_status dtl_environment_make_cpu_context(dtl_environment_t env,
                                             dtl_context_t* ctx) {
    if (!ctx) {
        return DTL_ERROR_NULL_POINTER;
    }

    if (!is_valid_environment(env)) {
        return DTL_ERROR_INVALID_ARGUMENT;
    }

    // Create C++ CPU-only context from environment
    auto cpp_ctx = env->impl->make_cpu_context();

    // Allocate C context handle
    dtl_context_s* impl = nullptr;
    try {
        impl = new dtl_context_s();
    } catch (...) {
        return DTL_ERROR_ALLOCATION_FAILED;
    }

    // CPU-only context: single process, no MPI
    impl->rank = cpp_ctx.rank();
    impl->size = cpp_ctx.size();
    impl->device_id = -1;
    impl->magic = dtl_context_s::VALID_MAGIC;
    impl->domain_flags = dtl_context_s::HAS_CPU;
    impl->error_handler = nullptr;
    impl->error_handler_user_data = nullptr;

    // No MPI for CPU-only context (intentional: this is for single-process use)

    *ctx = impl;
    return DTL_SUCCESS;
}

}  // extern "C"
