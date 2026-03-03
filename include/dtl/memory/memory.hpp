// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory.hpp
/// @brief Master include for DTL memory management module
/// @details Provides single-header access to all memory types.
/// @since 0.1.0

#pragma once

// Memory space base types
#include <dtl/memory/memory_space_base.hpp>

// Memory space implementations
#include <dtl/memory/host_memory_space.hpp>

// Allocators
#include <dtl/memory/allocator.hpp>
#include <dtl/memory/default_allocator.hpp>

namespace dtl {

// ============================================================================
// Memory Module Summary
// ============================================================================
//
// The memory module provides abstractions for memory allocation across
// different memory spaces (host, device, unified). It integrates with
// the backend concepts to support heterogeneous memory hierarchies.
//
// ============================================================================
// Memory Spaces
// ============================================================================
//
// Memory spaces represent different types of memory:
//
// - host_memory_space: Standard CPU heap memory (malloc/free)
// - pinned_memory_space: Page-locked host memory for fast GPU transfers
// - device_memory_space: GPU device memory (requires CUDA/HIP backend)
// - unified_memory_space: Managed memory accessible from host and device
//
// Each memory space satisfies the MemorySpace concept and provides:
// - allocate(size) / allocate(size, alignment)
// - deallocate(ptr, size)
// - properties() - Returns memory_space_properties
// - name() - Returns human-readable name
//
// ============================================================================
// Memory Space Properties
// ============================================================================
//
// memory_space_properties describes capabilities:
// - host_accessible: Can be read/written from CPU
// - device_accessible: Can be read/written from GPU
// - unified: Single address space for host and device
// - supports_atomics: Atomic operations available
// - pageable: Memory can be swapped (vs pinned)
// - alignment: Default alignment in bytes
//
// ============================================================================
// Allocators
// ============================================================================
//
// STL-compatible allocators using memory spaces:
//
// - memory_space_allocator<T, Space>: Stateless allocator for a space
// - polymorphic_allocator<T>: Type-erased allocator with runtime space
// - default_allocator<T>: Alias for host memory allocator
//
// All allocators satisfy std::allocator requirements and can be used
// with standard containers.
//
// ============================================================================
// Allocator Selection
// ============================================================================
//
// select_allocator<T, Placement> selects the appropriate allocator
// based on the placement policy:
//
// - host_only -> memory_space_allocator<T, host_memory_space>
// - device_only<Id> -> memory_space_allocator<T, device_memory_space>
// - unified_memory -> memory_space_allocator<T, unified_memory_space>
//
// ============================================================================
// Utilities
// ============================================================================
//
// Memory utilities:
// - is_aligned(ptr, alignment): Check pointer alignment
// - align_size(size, alignment): Round size up to alignment
// - zero_memory(ptr, size): Fill memory with zeros
// - copy_memory(dst, src, size): Copy memory
// - move_memory(dst, src, size): Move memory (handles overlap)
//
// Scoped allocation:
// - scoped_allocation<T>: RAII wrapper for allocated memory
// - make_scoped_allocation<T>(n): Create scoped allocation
//
// ============================================================================
// Usage Examples
// ============================================================================
//
// @code
// #include <dtl/memory/memory.hpp>
//
// // Use host memory space directly
// void* ptr = dtl::host_memory_space::allocate(1024);
// dtl::host_memory_space::deallocate(ptr, 1024);
//
// // Use STL-compatible allocator
// std::vector<double, dtl::default_allocator<double>> vec(100);
//
// // Use scoped allocation for exception safety
// auto buffer = dtl::make_scoped_allocation<float>(1000);
// for (size_t i = 0; i < buffer.size(); ++i) {
//     buffer[i] = static_cast<float>(i);
// }
// // Memory automatically freed when buffer goes out of scope
//
// // Check memory space properties
// constexpr auto props = dtl::host_memory_space::properties();
// static_assert(props.host_accessible);
// static_assert(!props.device_accessible);
// @endcode
//
// ============================================================================
// Integration with Containers
// ============================================================================
//
// DTL containers use allocators to control memory placement:
//
// @code
// // Vector with default (host) allocator
// dtl::distributed_vector<double> vec1(1000);
//
// // Vector with explicit placement policy
// dtl::distributed_vector<double, dtl::device_only<0>> vec2(1000);
//
// // The allocator is selected based on the placement policy
// @endcode
//
// ============================================================================

}  // namespace dtl
