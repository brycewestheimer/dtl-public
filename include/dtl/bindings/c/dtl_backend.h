// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file dtl_backend.h
/// @brief C ABI for DTL backend availability queries
/// @since 0.1.0

#ifndef DTL_BACKEND_H
#define DTL_BACKEND_H

#include <dtl/bindings/c/dtl_config.h>

DTL_C_BEGIN

/// @brief Get the name of the primary available backend
/// @return Backend name string ("MPI", "CUDA+MPI", "Single", etc.)
static inline const char* dtl_backend_name(void) {
    const int has_mpi = dtl_has_mpi();
    const int has_cuda = dtl_has_cuda();
    const int has_hip = dtl_has_hip();

    if (has_cuda && has_mpi) return "CUDA+MPI";
    if (has_hip && has_mpi) return "HIP+MPI";
    if (has_mpi) return "MPI";
    if (has_cuda) return "CUDA";
    if (has_hip) return "HIP";
    return "Single";
}

/// @brief Get the number of available backends
/// @return Count of enabled backends
static inline int dtl_backend_count(void) {
    return dtl_has_mpi() + dtl_has_cuda() + dtl_has_hip() + dtl_has_nccl() + dtl_has_shmem();
}

/// @brief DTL library version (convenience wrapper)
static inline const char* dtl_version(void) {
    return dtl_version_string();
}

DTL_C_END

#endif // DTL_BACKEND_H
