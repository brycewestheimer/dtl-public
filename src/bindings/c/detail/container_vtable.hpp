// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file container_vtable.hpp
 * @brief Type-erased container handle and vtable infrastructure for C ABI
 * @since 0.1.0
 *
 * This header defines the internal vtable structure used to implement
 * runtime policy dispatch in the C API. Each concrete C++ container
 * instantiation registers a vtable, and C handles store a pointer to
 * this vtable for all subsequent operations.
 */

#ifndef DTL_C_DETAIL_CONTAINER_VTABLE_HPP
#define DTL_C_DETAIL_CONTAINER_VTABLE_HPP

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>

#include <cstddef>
#include <cstdint>

namespace dtl::c::detail {

// ============================================================================
// Vtable Structures
// ============================================================================

/**
 * @brief Vtable for vector operations
 *
 * Contains function pointers for all operations that can be performed
 * on a vector. Each concrete instantiation (dtype × partition × placement)
 * provides its own vtable implementation.
 */
struct vector_vtable {
    // Lifecycle
    void (*destroy)(void* impl);

    // Size queries
    std::size_t (*global_size)(const void* impl);
    std::size_t (*local_size)(const void* impl);
    std::ptrdiff_t (*local_offset)(const void* impl);

    // Data access (returns nullptr for device-only placement)
    void* (*local_data_mut)(void* impl);
    const void* (*local_data)(const void* impl);

    // Device data access (returns nullptr for host placement)
    void* (*device_data_mut)(void* impl);
    const void* (*device_data)(const void* impl);

    // Copy helpers for device placement
    dtl_status (*copy_to_host)(const void* impl, void* host_buffer, std::size_t count);
    dtl_status (*copy_from_host)(void* impl, const void* host_buffer, std::size_t count);

    // Resize
    dtl_status (*resize)(void* impl, std::size_t new_size);

    // Fill (works on appropriate memory space)
    dtl_status (*fill)(void* impl, const void* value);

    // Distribution queries
    dtl_rank_t (*num_ranks)(const void* impl);
    dtl_rank_t (*rank)(const void* impl);
    dtl_rank_t (*owner)(const void* impl, dtl_index_t global_idx);
    int (*is_local)(const void* impl, dtl_index_t global_idx);
    dtl_index_t (*to_local)(const void* impl, dtl_index_t global_idx);
    dtl_index_t (*to_global)(const void* impl, dtl_index_t local_idx);

    // Algorithms (built-in operations that work on device)
    dtl_status (*reduce_sum)(const void* impl, void* result);
    dtl_status (*reduce_min)(const void* impl, void* result);
    dtl_status (*reduce_max)(const void* impl, void* result);

    // Sorting (ascending/descending)
    dtl_status (*sort_ascending)(void* impl);
    dtl_status (*sort_descending)(void* impl);
};

/**
 * @brief Vtable for array operations
 */
struct array_vtable {
    // Lifecycle
    void (*destroy)(void* impl);

    // Size queries
    std::size_t (*global_size)(const void* impl);
    std::size_t (*local_size)(const void* impl);
    std::ptrdiff_t (*local_offset)(const void* impl);

    // Data access
    void* (*local_data_mut)(void* impl);
    const void* (*local_data)(const void* impl);
    void* (*device_data_mut)(void* impl);
    const void* (*device_data)(const void* impl);

    // Copy helpers
    dtl_status (*copy_to_host)(const void* impl, void* host_buffer, std::size_t count);
    dtl_status (*copy_from_host)(void* impl, const void* host_buffer, std::size_t count);

    // Fill
    dtl_status (*fill)(void* impl, const void* value);

    // Distribution queries
    dtl_rank_t (*num_ranks)(const void* impl);
    dtl_rank_t (*rank)(const void* impl);
    dtl_rank_t (*owner)(const void* impl, dtl_index_t global_idx);
    int (*is_local)(const void* impl, dtl_index_t global_idx);
    dtl_index_t (*to_local)(const void* impl, dtl_index_t global_idx);
    dtl_index_t (*to_global)(const void* impl, dtl_index_t local_idx);

    // Algorithms
    dtl_status (*reduce_sum)(const void* impl, void* result);
    dtl_status (*reduce_min)(const void* impl, void* result);
    dtl_status (*reduce_max)(const void* impl, void* result);
};

/**
 * @brief Vtable for tensor operations
 */
struct tensor_vtable {
    // Lifecycle
    void (*destroy)(void* impl);

    // Size queries
    std::size_t (*global_size)(const void* impl);
    std::size_t (*local_size)(const void* impl);

    // Data access
    void* (*local_data_mut)(void* impl);
    const void* (*local_data)(const void* impl);
    void* (*device_data_mut)(void* impl);
    const void* (*device_data)(const void* impl);

    // Copy helpers
    dtl_status (*copy_to_host)(const void* impl, void* host_buffer, std::size_t count);
    dtl_status (*copy_from_host)(void* impl, const void* host_buffer, std::size_t count);

    // Fill
    dtl_status (*fill)(void* impl, const void* value);

    // Algorithms
    dtl_status (*reduce_sum)(const void* impl, void* result);

    // Shape queries
    int (*ndim)(const void* impl);
    void (*shape)(const void* impl, dtl_shape* out);
    void (*local_shape)(const void* impl, dtl_shape* out);
    dtl_size_t (*dim)(const void* impl, int dim);
    dtl_size_t (*stride)(const void* impl, int dim);
    int (*distributed_dim)(const void* impl);

    // Distribution queries
    dtl_rank_t (*num_ranks)(const void* impl);
    dtl_rank_t (*rank)(const void* impl);

    // ND access
    dtl_status (*get_local_nd)(const void* impl, const dtl_index_t* indices, void* value);
    dtl_status (*set_local_nd)(void* impl, const dtl_index_t* indices, const void* value);
    dtl_status (*get_local)(const void* impl, dtl_size_t linear_idx, void* value);
    dtl_status (*set_local)(void* impl, dtl_size_t linear_idx, const void* value);

    // Reshape
    dtl_status (*reshape)(void* impl, const dtl_shape* new_shape);
};

// ============================================================================
// Container Handle Metadata
// ============================================================================

/**
 * @brief Vtable for map operations
 *
 * Contains function pointers for all operations that can be performed
 * on a distributed map. Each concrete instantiation (key_dtype × value_dtype)
 * provides its own vtable implementation.
 */
struct map_vtable {
    // Lifecycle
    void (*destroy)(void* impl);

    // Size queries
    std::size_t (*local_size)(const void* impl);
    std::size_t (*global_size)(const void* impl);
    int (*local_empty)(const void* impl);

    // Access
    int (*contains_local)(const void* impl, const void* key);
    dtl_status (*find_local)(const void* impl, const void* key, void* value);
    dtl_rank_t (*owner)(const void* impl, const void* key);
    int (*is_local)(const void* impl, const void* key);

    // Modifiers
    dtl_status (*insert)(void* impl, const void* key, const void* value);
    dtl_status (*insert_or_assign)(void* impl, const void* key, const void* value);
    dtl_status (*erase)(void* impl, const void* key, std::size_t* erased);
    dtl_status (*clear)(void* impl);

    // Sync
    dtl_status (*flush)(void* impl);
    dtl_status (*sync)(void* impl);
    dtl_status (*barrier)(void* impl);

    // Dirty state
    int (*is_dirty)(const void* impl);
    int (*is_clean)(const void* impl);

    // Distribution
    dtl_rank_t (*num_ranks)(const void* impl);
    dtl_rank_t (*rank)(const void* impl);

    // Iterator
    dtl_status (*iter_begin)(void* impl, void** iter);
    dtl_status (*iter_next)(void* iter);
    dtl_status (*iter_key)(const void* iter, void* key);
    dtl_status (*iter_value)(const void* iter, void* value);
    void (*iter_destroy)(void* iter);
};

// ============================================================================
// Container Handle Metadata
// ============================================================================

/**
 * @brief Stored policy options for a container handle
 *
 * This mirrors dtl_container_options but is stored internally
 * with additional fields for consistency and error policies.
 */
struct stored_options {
    dtl_partition_policy partition;
    dtl_placement_policy placement;
    dtl_execution_policy execution;
    dtl_consistency_policy consistency;
    dtl_error_policy error;
    int device_id;
    std::size_t block_size;
};

/**
 * @brief Base container handle structure
 *
 * All C container handles contain this common header.
 */
struct container_handle_base {
    /// Validation magic number
    std::uint32_t magic;

    /// Data type
    dtl_dtype dtype;

    /// Creation context (must outlive the container)
    dtl_context_t ctx;

    /// Stored policy options
    stored_options options;
};

/**
 * @brief Vector handle internal structure
 */
struct vector_handle {
    /// Common header
    container_handle_base base;

    /// Vtable for operations
    const vector_vtable* vtable;

    /// Type-erased implementation pointer
    void* impl;

    /// Validation magic for vectors
    static constexpr std::uint32_t VALID_MAGIC = 0xCAFEF00D;
};

/**
 * @brief Array handle internal structure
 */
struct array_handle {
    /// Common header
    container_handle_base base;

    /// Vtable for operations
    const array_vtable* vtable;

    /// Type-erased implementation pointer
    void* impl;

    /// Validation magic for arrays
    static constexpr std::uint32_t VALID_MAGIC = 0xA22A7CAF;
};

/**
 * @brief Tensor handle internal structure
 */
struct tensor_handle {
    /// Common header
    container_handle_base base;

    /// Vtable for operations
    const tensor_vtable* vtable;

    /// Type-erased implementation pointer
    void* impl;

    /// Validation magic for tensors
    static constexpr std::uint32_t VALID_MAGIC = 0x7E350CAF;
};

// ============================================================================
// Validation Helpers
// ============================================================================

inline bool is_valid_vector_handle(const vector_handle* h) noexcept {
    return h && h->base.magic == vector_handle::VALID_MAGIC && h->base.ctx && h->vtable && h->impl;
}

inline bool is_valid_array_handle(const array_handle* h) noexcept {
    return h && h->base.magic == array_handle::VALID_MAGIC && h->base.ctx && h->vtable && h->impl;
}

inline bool is_valid_tensor_handle(const tensor_handle* h) noexcept {
    return h && h->base.magic == tensor_handle::VALID_MAGIC && h->base.ctx && h->vtable && h->impl;
}

// ============================================================================
// Map Handle
// ============================================================================

/**
 * @brief Map handle internal structure
 */
struct map_handle {
    /// Common header (key dtype stored in base.dtype)
    container_handle_base base;

    /// Value data type
    dtl_dtype value_dtype;

    /// Vtable for operations
    const map_vtable* vtable;

    /// Type-erased implementation pointer
    void* impl;

    /// Validation magic for maps
    static constexpr std::uint32_t VALID_MAGIC = 0xDA7A0AAF;
};

/**
 * @brief Map iterator handle internal structure
 */
struct map_iter_handle {
    /// Validation magic
    std::uint32_t magic;

    /// Parent map handle
    const map_handle* parent;

    /// Type-erased iterator implementation
    void* iter_impl;

    /// Validation magic for iterators
    static constexpr std::uint32_t VALID_MAGIC = 0x17E2A70F;
};

inline bool is_valid_map_handle(const map_handle* h) noexcept {
    return h && h->base.magic == map_handle::VALID_MAGIC && h->base.ctx && h->vtable && h->impl;
}

inline bool is_valid_map_iter_handle(const map_iter_handle* h) noexcept {
    return h && h->magic == map_iter_handle::VALID_MAGIC && h->parent && h->iter_impl;
}

// ============================================================================
// Policy Mapping
// ============================================================================

/**
 * @brief Check if a policy combination is supported
 * @param partition Partition policy
 * @param placement Placement policy
 * @param execution Execution policy
 * @param consistency Consistency policy
 * @param error Error policy
 * @return true if the combination is supported
 */
bool is_policy_combination_supported(
    dtl_partition_policy partition,
    dtl_placement_policy placement,
    dtl_execution_policy execution,
    dtl_consistency_policy consistency,
    dtl_error_policy error) noexcept;

/**
 * @brief Validate container options and normalize defaults
 * @param opts Input options (may be nullptr for defaults)
 * @param[out] out Normalized output options
 * @return DTL_SUCCESS or error code
 */
dtl_status validate_and_normalize_options(
    const dtl_container_options* opts,
    stored_options* out) noexcept;

}  // namespace dtl::c::detail

#endif /* DTL_C_DETAIL_CONTAINER_VTABLE_HPP */
