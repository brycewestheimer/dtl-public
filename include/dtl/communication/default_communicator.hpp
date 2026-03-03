// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file default_communicator.hpp
/// @brief Compile-time default communicator selection
/// @details Provides a type alias and factory function that resolve to
///          mpi::mpi_comm_adapter when MPI is enabled, or null_communicator
///          otherwise. This eliminates compile-time `DTL_ENABLE_MPI` guards from
///          user code.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/communication/communicator_base.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

namespace dtl {

// ============================================================================
// Default Communicator Type Alias
// ============================================================================

/// @brief Default communicator type, selected at compile time.
/// @details When DTL_ENABLE_MPI is defined and non-zero, this is
///          mpi::mpi_comm_adapter (wrapping MPI_COMM_WORLD). Otherwise,
///          it is null_communicator (rank 0, size 1, all ops are no-ops
///          or identity).
#if DTL_ENABLE_MPI
using default_communicator = mpi::mpi_comm_adapter;
#else
using default_communicator = null_communicator;
#endif

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Get the default world communicator.
/// @return mpi_comm_adapter wrapping MPI_COMM_WORLD (if MPI enabled),
///         or a null_communicator (single-rank, all ops are identity).
/// @deprecated Use environment::make_world_context() instead.
[[deprecated("Use environment::make_world_context()")]]
[[nodiscard]] inline default_communicator world_comm() {
    return default_communicator{};
}

}  // namespace dtl
