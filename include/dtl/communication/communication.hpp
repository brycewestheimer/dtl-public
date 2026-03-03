// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file communication.hpp
/// @brief Master include for DTL communication module
/// @details Provides single-header access to all communication operations.
/// @since 0.1.0

#pragma once

// Base communicator types
#include <dtl/communication/communicator_base.hpp>

// Communication operations
#include <dtl/communication/point_to_point.hpp>
#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/reduction_ops.hpp>

namespace dtl {

// ============================================================================
// Communication Module Summary
// ============================================================================
//
// The communication module provides abstractions for distributed communication
// following MPI-style semantics. All operations work with any type satisfying
// the Communicator, CollectiveCommunicator, or ReducingCommunicator concepts.
//
// ============================================================================
// Point-to-Point Operations
// ============================================================================
//
// Blocking:
// - send(comm, data, dest, tag)     - Send data to destination rank
// - recv(comm, data, source, tag)   - Receive data from source rank
// - ssend(comm, data, dest, tag)    - Synchronous send (blocks until received)
// - rsend(comm, data, dest, tag)    - Ready send (assumes recv posted)
// - sendrecv(comm, ...)             - Combined send and receive
//
// Non-blocking:
// - isend(comm, data, dest, tag)    - Non-blocking send, returns request
// - irecv(comm, data, source, tag)  - Non-blocking recv, returns request
//
// Request handling:
// - wait(comm, req)                 - Wait for request completion
// - test(comm, req)                 - Test if request complete
// - wait_all(comm, requests)        - Wait for all requests
// - wait_any(comm, requests)        - Wait for any request
//
// Probing:
// - probe(comm, source, tag)        - Blocking probe for message
// - iprobe(comm, source, tag)       - Non-blocking probe
//
// ============================================================================
// Collective Operations
// ============================================================================
//
// Synchronization:
// - barrier(comm)                   - Synchronize all ranks
//
// Data movement:
// - broadcast(comm, data, root)     - Broadcast from root to all
// - scatter(comm, send, recv, root) - Scatter from root to all
// - gather(comm, send, recv, root)  - Gather from all to root
// - allgather(comm, send, recv)     - Gather from all to all
// - alltoall(comm, send, recv)      - Exchange between all pairs
//
// Variable-count variants:
// - scatterv, gatherv, allgatherv, alltoallv
//
// ============================================================================
// Reduction Operations
// ============================================================================
//
// Reduce to root:
// - reduce(comm, send, recv, op, root)
// - sum(comm, send, recv, root)
//
// Reduce to all:
// - allreduce(comm, send, recv, op)
// - allreduce_inplace(comm, data, op)
// - allsum, allmax, allmin
//
// Prefix scans:
// - scan(comm, send, recv, op)      - Inclusive prefix scan
// - exscan(comm, send, recv, op)    - Exclusive prefix scan
//
// ============================================================================
// Reduction Operation Types
// ============================================================================
//
// Standard operations (all with identity element):
// - reduce_sum<T>      / plus       - Sum (identity: 0)
// - reduce_product<T>  / multiplies - Product (identity: 1)
// - reduce_min<T>      / minimum    - Minimum (identity: max)
// - reduce_max<T>      / maximum    - Maximum (identity: lowest)
// - reduce_land<T>                  - Logical AND (identity: true)
// - reduce_lor<T>                   - Logical OR (identity: false)
// - reduce_band<T>                  - Bitwise AND (identity: ~0)
// - reduce_bor<T>                   - Bitwise OR (identity: 0)
// - reduce_bxor<T>                  - Bitwise XOR (identity: 0)
//
// Location operations:
// - reduce_minloc<T,L>              - Min with location
// - reduce_maxloc<T,L>              - Max with location
//
// ============================================================================
// Communicator Types
// ============================================================================
//
// - null_communicator               - Single-process no-op communicator
// - communicator_handle             - Type-erased communicator wrapper
//
// ============================================================================
// Constants
// ============================================================================
//
// - any_tag                         - Match any message tag
// - any_source                      - Match any source rank
// - no_rank                         - Invalid rank sentinel
// - all_ranks                       - All ranks sentinel
//
// ============================================================================
// Usage Example
// ============================================================================
//
// @code
// #include <dtl/communication/communication.hpp>
//
// template <dtl::Communicator Comm>
// void example(Comm& comm) {
//     std::vector<double> data(100, 1.0);
//
//     // Collective sum reduction
//     std::vector<double> result(100);
//     dtl::allsum(comm, data, result);
//
//     // Point-to-point
//     if (comm.rank() == 0) {
//         dtl::send(comm, data, 1, 42);
//     } else if (comm.rank() == 1) {
//         dtl::recv(comm, data, 0, 42);
//     }
//
//     // Non-blocking
//     auto req = dtl::isend(comm, data, (comm.rank() + 1) % comm.size(), 0);
//     // ... do other work ...
//     dtl::wait(comm, req);
// }
// @endcode
//
// ============================================================================

}  // namespace dtl
