// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file fwd.hpp
/// @brief Forward declarations for all major DTL types
/// @details Provides forward declarations to minimize header dependencies
///          and enable type references without full definitions.
/// @since 0.1.0

#pragma once

#include <dtl/core/types.hpp>

#include <cstdint>
#include <functional>

namespace dtl {

// =============================================================================
// Error Types (Forward Declarations)
// =============================================================================

/// @brief Status code enumeration for operation results
enum class status_code : int;

/// @brief Status object representing operation outcome
class status;

/// @brief Result type template combining value or error
/// @tparam T The success value type
template <typename T>
class result;

// =============================================================================
// Policy Types (Forward Declarations)
// =============================================================================

// Partition policies
template <size_type N = 0>
struct block_partition;

template <size_type N = 0>
struct cyclic_partition;

template <typename Hash>
struct hash_partition;

struct replicated;

template <typename Fn>
struct custom_partition;

struct dynamic_block;

// Placement policies
struct host_only;

template <int DeviceId>
struct device_only;

struct device_preferred;
struct unified_memory;

template <typename PlacementMap>
struct explicit_placement;

// Consistency policies
struct bulk_synchronous;
struct sequential_consistent;
struct release_acquire;
struct relaxed;

// Execution policies
struct seq;
struct par;
struct async;

template <typename Stream>
struct on_stream;

// Error policies
struct expected_policy;
struct throwing_policy;
struct terminating_policy;

template <typename Handler>
struct callback_policy;

// =============================================================================
// View Types (Forward Declarations)
// =============================================================================

/// @brief Local view providing STL-compatible access to local partition
/// @tparam Container The distributed container type
template <typename Container>
class local_view;

/// @brief Global view providing distributed access (returns remote_ref for remote)
/// @tparam Container The distributed container type
template <typename Container>
class global_view;

/// @brief Segmented view for bulk distributed iteration
/// @tparam Container The distributed container type
template <typename Container>
class segmented_view;

/// @brief Explicit handle for remote element access
/// @tparam T The element type
/// @note No implicit conversions - "syntactically loud"
template <typename T>
class remote_ref;

/// @brief Subview of a distributed container
/// @tparam Container The distributed container type
template <typename Container>
class subview;

/// @brief Strided view with non-contiguous access pattern
/// @tparam Container The distributed container type
template <typename Container>
class strided_view;

// =============================================================================
// Container Types (Forward Declarations)
// =============================================================================

/// @brief 1D distributed vector container
/// @tparam T Element type
/// @tparam Policies Policy pack
template <typename T, typename... Policies>
class distributed_vector;

/// @brief N-dimensional distributed tensor container
/// @tparam T Element type
/// @tparam Rank Number of dimensions
/// @tparam Policies Policy pack
template <typename T, size_type Rank, typename... Policies>
class distributed_tensor;

/// @brief Non-owning distributed span view
/// @tparam T Element type
/// @tparam Extent Static extent (dynamic_extent for runtime size)
template <typename T, size_type Extent = dynamic_extent>
class distributed_span;

/// @brief Distributed associative map container
/// @tparam Key Key type
/// @tparam Value Value type
/// @tparam Hash Hash function type
/// @tparam KeyEqual Key equality function
/// @tparam Policies Policy pack
template <typename Key, typename Value,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          typename... Policies>
class distributed_map;

// =============================================================================
// Iterator Types (Forward Declarations)
// =============================================================================

/// @brief Iterator for local partition access
/// @tparam Container The distributed container type
template <typename Container>
class local_iterator;

/// @brief Iterator for global distributed access
/// @tparam Container The distributed container type
template <typename Container>
class global_iterator;

/// @brief Iterator for device memory access
/// @tparam Container The distributed container type
template <typename Container>
class device_iterator;

// =============================================================================
// Backend Concept Prototypes (Forward Declarations)
// =============================================================================

/// @brief Multi-domain context binding communication, execution, and memory domains
/// @tparam Domains... Domain types (mpi_domain, cpu_domain, cuda_domain, etc.)
/// @details Variadic template enabling type-safe access to multiple backend domains.
/// @since 0.1.0
template <typename... Domains>
class context;

// Domain forward declarations
struct mpi_domain_tag;
struct nccl_domain_tag;
struct shmem_domain_tag;
struct cpu_domain_tag;
struct cuda_domain_tag;
struct hip_domain_tag;

class mpi_domain;
class cpu_domain;
class cuda_domain;
class hip_domain;
class nccl_domain;
class shmem_domain;

// =============================================================================
// Async Types (Forward Declarations)
// =============================================================================

/// @brief Event handle for synchronization
class event;

}  // namespace dtl

namespace dtl::futures {

/// @brief Future for distributed async operations
/// @tparam T The result type
template <typename T>
class distributed_future;

}  // namespace dtl::futures

namespace dtl {

// Re-export futures types into dtl::
using futures::distributed_future;

// =============================================================================
// Communication Types (Forward Declarations)
// =============================================================================

/// @brief Base communicator interface
class communicator_base;

// =============================================================================
// Memory Types (Forward Declarations)
// =============================================================================

/// @brief Base memory space interface
class memory_space_base;

/// @brief Host (CPU) memory space
class host_memory_space;

/// @brief Distributed allocator
/// @tparam T The element type
template <typename T>
class distributed_allocator;

// =============================================================================
// MPMD Types (Forward Declarations)
// =============================================================================

}  // namespace dtl

namespace dtl::mpmd {

/// @brief Group of ranks for MPMD patterns
class rank_group;

/// @brief Manager for rank roles
class role_manager;

/// @brief Role enumeration for MPMD nodes
enum class node_role : std::uint32_t;

}  // namespace dtl::mpmd

namespace dtl {

// Re-export mpmd types into dtl::
using mpmd::rank_group;
using mpmd::role_manager;
using mpmd::node_role;

// =============================================================================
// Serialization Types (Forward Declarations)
// =============================================================================

/// @brief Serializer trait for custom type serialization
/// @tparam T The type to serialize
/// @tparam Enable SFINAE enabler (defaults defined in serializer.hpp)
template <typename T, typename Enable>
struct serializer;

// =============================================================================
// Index Types (Forward Declarations)
// =============================================================================

/// @brief Global index wrapper
/// @tparam T Underlying index type (default: index_t)
template <typename T = index_t>
class global_index;

/// @brief Local index wrapper
/// @tparam T Underlying index type (default: index_t)
template <typename T = index_t>
class local_index;

}  // namespace dtl
