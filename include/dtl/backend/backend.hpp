// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file backend.hpp
/// @brief Master include for DTL backend concepts and utilities
/// @details Provides single-header access to all backend abstractions.
/// @since 0.1.0

#pragma once

// Backend concepts
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/concepts/memory_transfer.hpp>
#include <dtl/backend/concepts/executor.hpp>
#include <dtl/backend/concepts/stream_executor.hpp>
#include <dtl/backend/concepts/serializer.hpp>
#include <dtl/backend/concepts/event.hpp>
#include <dtl/backend/concepts/distributed_future.hpp>
#include <dtl/backend/concepts/topology.hpp>

// Backend common utilities
#include <dtl/backend/common/backend_traits.hpp>
#include <dtl/backend/common/backend_context.hpp>

namespace dtl {

// ============================================================================
// Backend Module Summary
// ============================================================================
//
// The backend module defines C++20 concepts and traits for abstracting
// hardware and communication backends. This enables DTL to support multiple
// backends (MPI, CUDA, shared memory, etc.) through compile-time polymorphism.
//
// ============================================================================
// Core Concepts
// ============================================================================
//
// Communication:
// - Communicator: Point-to-point send/recv, non-blocking isend/irecv
// - CollectiveCommunicator: Barrier, broadcast, scatter, gather, alltoall
// - ReducingCommunicator: Reduce, allreduce operations
// - AsyncCommunicator: Non-blocking collective operations
//
// Memory:
// - MemorySpace: Allocate/deallocate memory with properties
// - TypedMemorySpace: Typed allocation with construct/destroy
// - AccessibleMemorySpace: Host/device accessibility queries
// - MemoryTransfer: Copy data between memory spaces
// - AsyncMemoryTransfer: Non-blocking memory copies
//
// Execution:
// - Executor: Basic work execution (execute callable)
// - SyncExecutor: Guarantees synchronous completion
// - ParallelExecutor: Parallel for, concurrency queries
// - BulkExecutor: Chunked bulk operations
// - StreamExecutor: GPU-style async streams
// - MultiStreamExecutor: Multiple concurrent streams
//
// Synchronization:
// - Event: Wait, query status, synchronize
// - TimedEvent: Timeout support, elapsed time
// - RecordableEvent: Record on stream
//
// Async Results:
// - DistributedFuture: Get, wait, valid
// - PollableFuture: Non-blocking is_ready
// - TimedFuture: wait_for with timeout
// - ContinuableFuture: .then() continuations
//
// Serialization:
// - Serializer: serialized_size, serialize, deserialize
// - FixedSizeSerializer: Compile-time known size
// - TriviallySerializable: Trivially copyable types
//
// Topology:
// - Topology: Node counts, hostname
// - NumaTopology: NUMA node queries
// - GpuTopology: GPU device queries
// - NetworkTopology: Inter-rank distance/bandwidth
//
// ============================================================================
// Traits and Utilities
// ============================================================================
//
// backend_traits<T>: Query backend capabilities at compile time
// - supports_point_to_point, supports_collectives, supports_gpu_aware
// - supports_async, supports_thread_multiple, supports_rdma
//
// combined_backend_traits<Ts...>: Combine multiple backend traits
//
// backend_context<T>: RAII lifecycle management for backends
// scoped_context<T>: Automatic init/finalize guard
//
// ============================================================================
// Standard Implementations
// ============================================================================
//
// Executors:
// - inline_executor: Immediate execution in calling thread
// - sequential_executor: Sequential parallel_for
// - host_stream_executor: Host-side stream simulation
//
// Events:
// - null_event: No-op event (always complete)
//
// Futures:
// - ready_future<T>: Immediately ready future
// - failed_future<T>: Immediately failed future
//
// Topology:
// - basic_topology: Basic system queries
//
// Serializers:
// - trivial_serializer<T>: For trivially copyable types
//
// ============================================================================
// Backend Tags
// ============================================================================
//
// Communication backends:
// - mpi_communicator_tag
// - shared_memory_communicator_tag
// - gpu_communicator_tag (NCCL-style)
//
// Memory space tags:
// - host_memory_space_tag
// - device_memory_space_tag
// - unified_memory_space_tag
// - pinned_memory_space_tag
//
// Executor tags:
// - inline_executor_tag
// - thread_pool_executor_tag
// - single_thread_executor_tag
// - gpu_executor_tag
//
// Stream tags:
// - cuda_stream_tag
// - hip_stream_tag
// - sycl_queue_tag
// - host_stream_tag
//
// Event tags:
// - host_event_tag
// - cuda_event_tag
// - hip_event_tag
// - sycl_event_tag
//
// Backend feature tags:
// - mpi_backend_tag
// - cuda_backend_tag
// - hip_backend_tag
// - sycl_backend_tag
// - nccl_backend_tag
// - shared_memory_backend_tag
// - shmem_backend_tag
//
// ============================================================================
// Usage Example
// ============================================================================
//
// @code
// #include <dtl/backend/backend.hpp>
//
// // Check backend capabilities at compile time
// static_assert(dtl::backend_traits<dtl::mpi_backend_tag>::supports_collectives);
//
// // Use concepts to constrain templates
// template <dtl::Communicator Comm>
// void distributed_work(Comm& comm) {
//     if (comm.rank() == 0) {
//         comm.send(data, size, 1, tag);
//     }
// }
//
// // Use standard executors
// dtl::inline_executor exec;
// exec.execute([]{ compute(); });
//
// // Manage backend lifetime
// auto ctx = dtl::make_scoped_context<dtl::mpi_backend_tag>();
// @endcode
//
// ============================================================================

}  // namespace dtl
