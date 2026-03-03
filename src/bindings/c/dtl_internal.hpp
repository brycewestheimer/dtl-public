// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file dtl_internal.hpp
 * @brief DTL C bindings - Internal shared structures
 * @since 0.1.0
 *
 * This header contains internal structure definitions shared across
 * C binding implementation files. Not part of the public API.
 */

#ifndef DTL_INTERNAL_HPP
#define DTL_INTERNAL_HPP

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_environment.h>
#include <dtl/core/types.hpp>
#include <cstdint>
#include <atomic>
#include <limits>
#include <memory>

#ifdef DTL_HAS_MPI
#include <mpi.h>
#endif

// Forward declare dtl::environment to avoid pulling in heavy headers
namespace dtl {
class environment;
}  // namespace dtl

// ============================================================================
// Environment Internal Structure
// ============================================================================

/**
 * Environment implementation
 *
 * Wraps a dtl::environment instance with a magic number for handle validation.
 * The dtl::environment is reference-counted internally; each dtl_environment_s
 * holds one reference via its raw pointer (manually deleted in
 * dtl_environment_destroy). A raw pointer is used instead of unique_ptr to
 * avoid requiring the complete dtl::environment type in translation units
 * that include this header but do not manage environment objects.
 */
struct dtl_environment_s {
    dtl::environment* impl;  // Actual C++ environment object (owned, manually deleted)
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xCA11AB1E;
};

// ============================================================================
// Context Internal Structure
// ============================================================================

/**
 * Context implementation
 */
struct dtl_context_s {
    dtl_rank_t rank;
    dtl_rank_t size;
#ifdef DTL_HAS_MPI
    MPI_Comm comm;
    bool owns_comm;
    bool initialized_mpi;
    bool finalize_mpi;
#endif
#ifdef DTL_HAS_NCCL
    void* nccl_comm;  // ncclComm_t (stored as void* to avoid header dep)
    void* cuda_stream; // cudaStream_t for NCCL operations
#endif
    int device_id;
    int determinism_mode;
    int reduction_schedule_policy;
    int progress_ordering_policy;
    uint32_t domain_flags;
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xDEADBEEF;

    // Error handler state (per-context)
    // Used when container options specify DTL_ERROR_POLICY_CALLBACK.
    void (*error_handler)(dtl_status status, const char* message, void* user_data);
    void* error_handler_user_data;

    // Domain flag bits
    static constexpr uint32_t HAS_MPI   = 0x01;
    static constexpr uint32_t HAS_CUDA  = 0x02;
    static constexpr uint32_t HAS_NCCL  = 0x04;
    static constexpr uint32_t HAS_SHMEM = 0x08;
    static constexpr uint32_t HAS_CPU   = 0x10;
};

// ============================================================================
// Request Internal Structure
// ============================================================================

/**
 * Async request implementation
 */
struct dtl_request_s {
    struct async_state {
        std::atomic<bool> completed{false};
        std::atomic<bool> cancelled{false};
    };

#ifdef DTL_HAS_MPI
    MPI_Request mpi_request;
    bool is_mpi_request;  // True if mpi_request is valid
#endif
    std::shared_ptr<async_state> state;
    dtl::size_type progress_callback_id = static_cast<dtl::size_type>(-1);
    uint32_t magic;
    static constexpr uint32_t VALID_MAGIC = 0xBEEFCAFE;
};

// ============================================================================
// Validation Helpers
// ============================================================================

inline bool is_valid_environment(dtl_environment_t env) {
    return env && env->magic == dtl_environment_s::VALID_MAGIC;
}

inline bool is_valid_context(dtl_context_t ctx) {
    return ctx && ctx->magic == dtl_context_s::VALID_MAGIC;
}

inline bool is_valid_request(dtl_request_t req) {
    return req && req->magic == dtl_request_s::VALID_MAGIC;
}

inline bool checked_size_to_int(dtl_size_t value, int* out) noexcept {
    if (!out) {
        return false;
    }
    constexpr auto max_int = static_cast<dtl_size_t>(std::numeric_limits<int>::max());
    if (value > max_int) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

#endif /* DTL_INTERNAL_HPP */
