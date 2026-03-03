// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file rma.hpp
/// @brief Umbrella header for DTL RMA (Remote Memory Access) support
/// @details Provides one-sided communication operations for efficient
///          distributed data access.
/// @since 0.1.0

#pragma once

// Core RMA concepts and types
#include <dtl/backend/concepts/rma_communicator.hpp>

// Memory window abstraction
#include <dtl/communication/memory_window.hpp>

// RMA operations
#include <dtl/communication/rma_operations.hpp>
#include <dtl/communication/rma_atomic.hpp>

// RAII synchronization guards
#include <dtl/rma/window_guard.hpp>

// Async RMA with progress engine integration
#include <dtl/rma/async_rma.hpp>

// Integration with dtl::remote
#include <dtl/rma/remote_integration.hpp>

/// @namespace dtl::rma
/// @brief RMA (Remote Memory Access) operations for one-sided communication
///
/// @details The dtl::rma namespace provides one-sided communication primitives
/// that allow direct access to remote memory without target process participation.
///
/// @par Key Components:
/// - **memory_window**: RAII memory window abstraction
/// - **put/get**: One-sided data transfer operations
/// - **accumulate**: Atomic remote accumulation
/// - **fence/lock**: Synchronization mechanisms
/// - **async_put/async_get**: Progress-engine integrated async operations
/// - **rma_remote_ref**: RMA-backed remote element access
///
/// @par Usage Example:
/// @code
/// // Create a memory window
/// std::vector<int> data(1000);
/// auto win_result = memory_window::create(data.data(), data.size() * sizeof(int));
/// auto& window = *win_result;
///
/// // Active-target synchronization with fence
/// window.fence();
/// rma::put(target_rank, offset, std::span{data_to_send}, window);
/// window.fence();
///
/// // Passive-target with lock/unlock
/// {
///     rma::lock_guard lock(target_rank, window);
///     rma::put(target_rank, offset, std::span{data_to_send}, window);
/// }  // Automatically unlocks
///
/// // Async operations
/// auto put_op = rma::async_put_to(target_rank, offset, std::span{data}, window);
/// // ... do other work ...
/// put_op.wait();
/// @endcode
///
/// @par MPI Backend:
/// When DTL_ENABLE_MPI is defined, the MPI-3 RMA backend is available:
/// - `backends/mpi/mpi_window.hpp` - MPI_Win wrapper
/// - `backends/mpi/mpi_rma_adapter.hpp` - RmaCommunicator implementation
///
/// @see dtl::memory_window
/// @see dtl::rma::put
/// @see dtl::rma::get
/// @see dtl::rma::async_put
/// @see dtl::rma::rma_remote_ref

namespace dtl::rma {

// Re-export memory_window into dtl::rma:: for discoverability
// (memory_window is defined in dtl:: via communication/memory_window.hpp)
using dtl::memory_window;

}  // namespace dtl::rma
