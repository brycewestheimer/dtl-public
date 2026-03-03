// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file array_dispatch.hpp
 * @brief Array creation dispatch for C ABI policy selection
 * @since 0.1.0
 *
 * This header implements create-time dispatch for arrays based on
 * runtime policy options. Arrays share most logic with vectors.
 */

#ifndef DTL_C_DETAIL_ARRAY_DISPATCH_HPP
#define DTL_C_DETAIL_ARRAY_DISPATCH_HPP

#include "container_vtable.hpp"
#include "policy_matrix.hpp"
#include "vector_dispatch.hpp"

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>

namespace dtl::c::detail {

// ============================================================================
// Array Implementation
// ============================================================================

/**
 * @brief Concrete array implementation for host placement
 * @tparam T Element type
 *
 * Arrays are similar to vectors but have fixed size after creation.
 * We reuse the host_vector_impl for storage.
 */
template <typename T>
class host_array_impl : public host_vector_impl<T> {
public:
    using host_vector_impl<T>::host_vector_impl;

    // Arrays cannot be resized
    dtl_status resize(std::size_t /*new_size*/) {
        return DTL_ERROR_NOT_SUPPORTED;  // Arrays have fixed size
    }
};

// ============================================================================
// Vtable Factory Functions
// ============================================================================

/**
 * @brief Create a vtable for a host array implementation
 */
template <typename T>
const array_vtable* get_host_array_vtable() noexcept {
    static const array_vtable vtable = {
        // destroy
        [](void* impl) {
            delete static_cast<host_array_impl<T>*>(impl);
        },

        // global_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_array_impl<T>*>(impl)->global_size();
        },

        // local_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_array_impl<T>*>(impl)->local_size();
        },

        // local_offset
        [](const void* impl) -> std::ptrdiff_t {
            return static_cast<const host_array_impl<T>*>(impl)->local_offset();
        },

        // local_data_mut
        [](void* impl) -> void* {
            return static_cast<host_array_impl<T>*>(impl)->data();
        },

        // local_data
        [](const void* impl) -> const void* {
            return static_cast<const host_array_impl<T>*>(impl)->data();
        },

        // device_data_mut
        [](void* /*impl*/) -> void* { return nullptr; },

        // device_data
        [](const void* /*impl*/) -> const void* { return nullptr; },

        // copy_to_host
        [](const void* impl, void* host_buffer, std::size_t count) -> dtl_status {
            const auto* arr = static_cast<const host_array_impl<T>*>(impl);
            if (count > arr->local_size()) count = arr->local_size();
            std::memcpy(host_buffer, arr->data(), count * sizeof(T));
            return DTL_SUCCESS;
        },

        // copy_from_host
        [](void* impl, const void* host_buffer, std::size_t count) -> dtl_status {
            auto* arr = static_cast<host_array_impl<T>*>(impl);
            if (count > arr->local_size()) count = arr->local_size();
            std::memcpy(arr->data(), host_buffer, count * sizeof(T));
            return DTL_SUCCESS;
        },

        // fill
        [](void* impl, const void* value) -> dtl_status {
            return static_cast<host_array_impl<T>*>(impl)->fill(*static_cast<const T*>(value));
        },

        // num_ranks
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_array_impl<T>*>(impl)->num_ranks();
        },

        // rank
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_array_impl<T>*>(impl)->rank();
        },

        // owner
        [](const void* impl, dtl_index_t global_idx) -> dtl_rank_t {
            return static_cast<const host_array_impl<T>*>(impl)->owner(global_idx);
        },

        // is_local
        [](const void* impl, dtl_index_t global_idx) -> int {
            return static_cast<const host_array_impl<T>*>(impl)->is_local(global_idx) ? 1 : 0;
        },

        // to_local
        [](const void* impl, dtl_index_t global_idx) -> dtl_index_t {
            return static_cast<const host_array_impl<T>*>(impl)->to_local(global_idx);
        },

        // to_global
        [](const void* impl, dtl_index_t local_idx) -> dtl_index_t {
            return static_cast<const host_array_impl<T>*>(impl)->to_global(local_idx);
        },

        // reduce_sum
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_array_impl<T>*>(impl)->reduce_sum();
            return DTL_SUCCESS;
        },

        // reduce_min
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_array_impl<T>*>(impl)->reduce_min();
            return DTL_SUCCESS;
        },

        // reduce_max
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_array_impl<T>*>(impl)->reduce_max();
            return DTL_SUCCESS;
        }
    };
    return &vtable;
}

// ============================================================================
// Dispatch Function
// ============================================================================

/**
 * @brief Create an array implementation based on options
 */
inline dtl_status dispatch_create_array(
    dtl_dtype dtype,
    std::size_t size,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    const stored_options& opts,
    const array_vtable** out_vtable,
    void** out_impl) {

    // For device placements, currently return NOT_SUPPORTED
    if (opts.placement == DTL_PLACEMENT_DEVICE ||
        opts.placement == DTL_PLACEMENT_DEVICE_PREFERRED) {
#if !DTL_ENABLE_CUDA
        return DTL_ERROR_NOT_SUPPORTED;
#endif
    }

    try {
        switch (dtype) {
            case DTL_DTYPE_INT8: {
                *out_vtable = get_host_array_vtable<int8_t>();
                *out_impl = new host_array_impl<int8_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT16: {
                *out_vtable = get_host_array_vtable<int16_t>();
                *out_impl = new host_array_impl<int16_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT32: {
                *out_vtable = get_host_array_vtable<int32_t>();
                *out_impl = new host_array_impl<int32_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT64: {
                *out_vtable = get_host_array_vtable<int64_t>();
                *out_impl = new host_array_impl<int64_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT8:
            case DTL_DTYPE_BYTE:
            case DTL_DTYPE_BOOL: {
                *out_vtable = get_host_array_vtable<uint8_t>();
                *out_impl = new host_array_impl<uint8_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT16: {
                *out_vtable = get_host_array_vtable<uint16_t>();
                *out_impl = new host_array_impl<uint16_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT32: {
                *out_vtable = get_host_array_vtable<uint32_t>();
                *out_impl = new host_array_impl<uint32_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT64: {
                *out_vtable = get_host_array_vtable<uint64_t>();
                *out_impl = new host_array_impl<uint64_t>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_FLOAT32: {
                *out_vtable = get_host_array_vtable<float>();
                *out_impl = new host_array_impl<float>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_FLOAT64: {
                *out_vtable = get_host_array_vtable<double>();
                *out_impl = new host_array_impl<double>(size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            default:
                return DTL_ERROR_INVALID_ARGUMENT;
        }
    } catch (const std::bad_alloc&) {
        return DTL_ERROR_ALLOCATION_FAILED;
    } catch (...) {
        return DTL_ERROR_UNKNOWN;
    }

    return DTL_SUCCESS;
}

}  // namespace dtl::c::detail

#endif /* DTL_C_DETAIL_ARRAY_DISPATCH_HPP */
