// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file vector_dispatch.hpp
 * @brief Vector creation dispatch for C ABI policy selection
 * @since 0.1.0
 *
 * This header implements create-time dispatch for vectors based on
 * runtime policy options. It instantiates the supported matrix of
 * {dtype × partition × placement × execution} combinations.
 */

#ifndef DTL_C_DETAIL_VECTOR_DISPATCH_HPP
#define DTL_C_DETAIL_VECTOR_DISPATCH_HPP

#include "container_vtable.hpp"
#include "policy_matrix.hpp"

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>
#include <dtl/bindings/c/dtl_context.h>

#include "placement_mapping.hpp"

#include <algorithm>
#include <cstring>
#include <new>
#include <vector>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::c::detail {

// ============================================================================
// Host-Based Vector Implementation
// ============================================================================

/**
 * @brief Concrete vector implementation for host placement
 * @tparam T Element type
 *
 * This implementation stores data in a std::vector on the host.
 * It supports block, cyclic, and replicated partitions.
 */
template <typename T>
class host_vector_impl {
public:
    host_vector_impl(std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
                     dtl_partition_policy partition, std::size_t block_size = 1)
        : global_size_(global_size)
        , rank_(rank)
        , num_ranks_(num_ranks)
        , partition_(partition)
        , block_size_(block_size) {

        compute_local_partition();
        storage_.resize(local_size_);
    }

    ~host_vector_impl() = default;

    // Size queries
    std::size_t global_size() const noexcept { return global_size_; }
    std::size_t local_size() const noexcept { return local_size_; }
    std::ptrdiff_t local_offset() const noexcept { return local_offset_; }

    // Data access
    T* data() noexcept { return storage_.data(); }
    const T* data() const noexcept { return storage_.data(); }

    // Distribution
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return rank_; }

    dtl_rank_t owner(dtl_index_t global_idx) const noexcept {
        if (global_idx < 0 || static_cast<std::size_t>(global_idx) >= global_size_) {
            return DTL_NO_RANK;
        }

        switch (partition_) {
            case DTL_PARTITION_BLOCK:
                return owner_block(global_idx);
            case DTL_PARTITION_CYCLIC:
                return owner_cyclic(global_idx);
            case DTL_PARTITION_REPLICATED:
                return rank_;  // All ranks own all data
            case DTL_PARTITION_BLOCK_CYCLIC:
                return owner_block_cyclic(global_idx);
            default:
                return DTL_NO_RANK;
        }
    }

    bool is_local(dtl_index_t global_idx) const noexcept {
        if (partition_ == DTL_PARTITION_REPLICATED) {
            return global_idx >= 0 && static_cast<std::size_t>(global_idx) < global_size_;
        }
        return owner(global_idx) == rank_;
    }

    dtl_index_t to_local(dtl_index_t global_idx) const noexcept {
        if (!is_local(global_idx)) return -1;

        switch (partition_) {
            case DTL_PARTITION_BLOCK:
                return global_idx - local_offset_;
            case DTL_PARTITION_CYCLIC:
                return global_idx / num_ranks_;
            case DTL_PARTITION_REPLICATED:
                return global_idx;
            case DTL_PARTITION_BLOCK_CYCLIC:
                return to_local_block_cyclic(global_idx);
            default:
                return -1;
        }
    }

    dtl_index_t to_global(dtl_index_t local_idx) const noexcept {
        if (local_idx < 0 || static_cast<std::size_t>(local_idx) >= local_size_) {
            return -1;
        }

        switch (partition_) {
            case DTL_PARTITION_BLOCK:
                return local_offset_ + local_idx;
            case DTL_PARTITION_CYCLIC:
                return local_idx * num_ranks_ + rank_;
            case DTL_PARTITION_REPLICATED:
                return local_idx;
            case DTL_PARTITION_BLOCK_CYCLIC:
                return to_global_block_cyclic(local_idx);
            default:
                return -1;
        }
    }

    // Resize
    dtl_status resize(std::size_t new_global_size) {
        global_size_ = new_global_size;
        compute_local_partition();
        try {
            storage_.resize(local_size_);
        } catch (...) {
            return DTL_ERROR_ALLOCATION_FAILED;
        }
        return DTL_SUCCESS;
    }

    // Fill
    dtl_status fill(const T& value) {
        std::fill(storage_.begin(), storage_.end(), value);
        return DTL_SUCCESS;
    }

    // Reductions
    T reduce_sum() const {
        T sum{};
        for (const auto& val : storage_) {
            sum = sum + val;
        }
        return sum;
    }

    T reduce_min() const {
        if (storage_.empty()) return T{};
        return *std::min_element(storage_.begin(), storage_.end());
    }

    T reduce_max() const {
        if (storage_.empty()) return T{};
        return *std::max_element(storage_.begin(), storage_.end());
    }

    // Sorting
    void sort_ascending() {
        std::sort(storage_.begin(), storage_.end());
    }

    void sort_descending() {
        std::sort(storage_.begin(), storage_.end(), std::greater<T>());
    }

private:
    void compute_local_partition() {
        switch (partition_) {
            case DTL_PARTITION_BLOCK:
                compute_block_partition();
                break;
            case DTL_PARTITION_CYCLIC:
                compute_cyclic_partition();
                break;
            case DTL_PARTITION_REPLICATED:
                local_size_ = global_size_;
                local_offset_ = 0;
                break;
            case DTL_PARTITION_BLOCK_CYCLIC:
                compute_block_cyclic_partition();
                break;
            default:
                compute_block_partition();
                break;
        }
    }

    void compute_block_partition() {
        std::size_t base = global_size_ / num_ranks_;
        std::size_t remainder = global_size_ % num_ranks_;

        if (static_cast<std::size_t>(rank_) < remainder) {
            local_size_ = base + 1;
            local_offset_ = rank_ * (base + 1);
        } else {
            local_size_ = base;
            local_offset_ = remainder * (base + 1) + (rank_ - remainder) * base;
        }
    }

    void compute_cyclic_partition() {
        std::size_t base = global_size_ / num_ranks_;
        std::size_t remainder = global_size_ % num_ranks_;
        local_size_ = base + (static_cast<std::size_t>(rank_) < remainder ? 1 : 0);
        // Cyclic doesn't have contiguous offset
        local_offset_ = rank_;
    }

    void compute_block_cyclic_partition() {
        // Block-cyclic: blocks of block_size_ distributed cyclically
        std::size_t total_blocks = (global_size_ + block_size_ - 1) / block_size_;
        std::size_t blocks_per_rank = total_blocks / num_ranks_;
        std::size_t extra_blocks = total_blocks % num_ranks_;

        std::size_t my_blocks = blocks_per_rank + (static_cast<std::size_t>(rank_) < extra_blocks ? 1 : 0);

        // Calculate exact local size accounting for partial last block
        local_size_ = 0;
        for (std::size_t b = 0; b < my_blocks; ++b) {
            std::size_t block_idx = b * num_ranks_ + rank_;
            std::size_t block_start = block_idx * block_size_;
            std::size_t block_end = std::min(block_start + block_size_, global_size_);
            if (block_start < global_size_) {
                local_size_ += block_end - block_start;
            }
        }

        // Block-cyclic doesn't have a single contiguous offset
        local_offset_ = rank_ * block_size_;
    }

    dtl_rank_t owner_block(dtl_index_t global_idx) const noexcept {
        std::size_t base = global_size_ / num_ranks_;
        std::size_t remainder = global_size_ % num_ranks_;
        std::size_t boundary = remainder * (base + 1);

        if (static_cast<std::size_t>(global_idx) < boundary) {
            return static_cast<dtl_rank_t>(global_idx / (base + 1));
        } else {
            return static_cast<dtl_rank_t>(remainder + (global_idx - boundary) / base);
        }
    }

    dtl_rank_t owner_cyclic(dtl_index_t global_idx) const noexcept {
        return static_cast<dtl_rank_t>(global_idx % num_ranks_);
    }

    dtl_rank_t owner_block_cyclic(dtl_index_t global_idx) const noexcept {
        std::size_t block_idx = global_idx / block_size_;
        return static_cast<dtl_rank_t>(block_idx % num_ranks_);
    }

    dtl_index_t to_local_block_cyclic(dtl_index_t global_idx) const noexcept {
        std::size_t block_idx = global_idx / block_size_;
        std::size_t offset_in_block = global_idx % block_size_;
        std::size_t local_block = block_idx / num_ranks_;
        return static_cast<dtl_index_t>(local_block * block_size_ + offset_in_block);
    }

    dtl_index_t to_global_block_cyclic(dtl_index_t local_idx) const noexcept {
        std::size_t local_block = local_idx / block_size_;
        std::size_t offset_in_block = local_idx % block_size_;
        std::size_t global_block = local_block * num_ranks_ + rank_;
        return static_cast<dtl_index_t>(global_block * block_size_ + offset_in_block);
    }

    std::vector<T> storage_;
    std::size_t global_size_;
    std::size_t local_size_;
    std::ptrdiff_t local_offset_;
    dtl_rank_t rank_;
    dtl_rank_t num_ranks_;
    dtl_partition_policy partition_;
    std::size_t block_size_;
};

// ============================================================================
// Vtable Factory Functions
// ============================================================================

/**
 * @brief Create a vtable for a host vector implementation
 */
template <typename T>
const vector_vtable* get_host_vector_vtable() noexcept {
    static const vector_vtable vtable = {
        // destroy
        [](void* impl) {
            delete static_cast<host_vector_impl<T>*>(impl);
        },

        // global_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_vector_impl<T>*>(impl)->global_size();
        },

        // local_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_vector_impl<T>*>(impl)->local_size();
        },

        // local_offset
        [](const void* impl) -> std::ptrdiff_t {
            return static_cast<const host_vector_impl<T>*>(impl)->local_offset();
        },

        // local_data_mut
        [](void* impl) -> void* {
            return static_cast<host_vector_impl<T>*>(impl)->data();
        },

        // local_data
        [](const void* impl) -> const void* {
            return static_cast<const host_vector_impl<T>*>(impl)->data();
        },

        // device_data_mut (not available for host)
        [](void* /*impl*/) -> void* { return nullptr; },

        // device_data (not available for host)
        [](const void* /*impl*/) -> const void* { return nullptr; },

        // copy_to_host (already on host, just memcpy)
        [](const void* impl, void* host_buffer, std::size_t count) -> dtl_status {
            const auto* vec = static_cast<const host_vector_impl<T>*>(impl);
            if (count > vec->local_size()) count = vec->local_size();
            std::memcpy(host_buffer, vec->data(), count * sizeof(T));
            return DTL_SUCCESS;
        },

        // copy_from_host (already on host, just memcpy)
        [](void* impl, const void* host_buffer, std::size_t count) -> dtl_status {
            auto* vec = static_cast<host_vector_impl<T>*>(impl);
            if (count > vec->local_size()) count = vec->local_size();
            std::memcpy(vec->data(), host_buffer, count * sizeof(T));
            return DTL_SUCCESS;
        },

        // resize
        [](void* impl, std::size_t new_size) -> dtl_status {
            return static_cast<host_vector_impl<T>*>(impl)->resize(new_size);
        },

        // fill
        [](void* impl, const void* value) -> dtl_status {
            return static_cast<host_vector_impl<T>*>(impl)->fill(*static_cast<const T*>(value));
        },

        // num_ranks
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_vector_impl<T>*>(impl)->num_ranks();
        },

        // rank
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_vector_impl<T>*>(impl)->rank();
        },

        // owner
        [](const void* impl, dtl_index_t global_idx) -> dtl_rank_t {
            return static_cast<const host_vector_impl<T>*>(impl)->owner(global_idx);
        },

        // is_local
        [](const void* impl, dtl_index_t global_idx) -> int {
            return static_cast<const host_vector_impl<T>*>(impl)->is_local(global_idx) ? 1 : 0;
        },

        // to_local
        [](const void* impl, dtl_index_t global_idx) -> dtl_index_t {
            return static_cast<const host_vector_impl<T>*>(impl)->to_local(global_idx);
        },

        // to_global
        [](const void* impl, dtl_index_t local_idx) -> dtl_index_t {
            return static_cast<const host_vector_impl<T>*>(impl)->to_global(local_idx);
        },

        // reduce_sum
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_vector_impl<T>*>(impl)->reduce_sum();
            return DTL_SUCCESS;
        },

        // reduce_min
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_vector_impl<T>*>(impl)->reduce_min();
            return DTL_SUCCESS;
        },

        // reduce_max
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_vector_impl<T>*>(impl)->reduce_max();
            return DTL_SUCCESS;
        },

        // sort_ascending
        [](void* impl) -> dtl_status {
            static_cast<host_vector_impl<T>*>(impl)->sort_ascending();
            return DTL_SUCCESS;
        },

        // sort_descending
        [](void* impl) -> dtl_status {
            static_cast<host_vector_impl<T>*>(impl)->sort_descending();
            return DTL_SUCCESS;
        }
    };
    return &vtable;
}

// ============================================================================
// CUDA Device Vector Implementation
// ============================================================================

#if DTL_ENABLE_CUDA

/**
 * @brief Concrete vector implementation for CUDA device placement
 * @tparam T Element type (must be trivially_copyable)
 *
 * Stores data in CUDA device memory allocated via cudaMalloc.
 * Host access is not available — local_data() returns nullptr.
 * Use copy_to_host/copy_from_host for data transfer.
 */
template <typename T>
class device_vector_impl {
public:
    device_vector_impl(std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
                       dtl_partition_policy partition, int device_id,
                       std::size_t block_size = 1)
        : global_size_(global_size)
        , rank_(rank)
        , num_ranks_(num_ranks)
        , partition_(partition)
        , block_size_(block_size)
        , device_id_(device_id)
        , device_ptr_(nullptr) {

        compute_local_partition();

        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (local_size_ > 0) {
            cudaMalloc(&device_ptr_, local_size_ * sizeof(T));
            cudaMemset(device_ptr_, 0, local_size_ * sizeof(T));
        }
    }

    ~device_vector_impl() {
        if (device_ptr_) {
            if (device_id_ >= 0) cudaSetDevice(device_id_);
            cudaFree(device_ptr_);
        }
    }

    device_vector_impl(const device_vector_impl&) = delete;
    device_vector_impl& operator=(const device_vector_impl&) = delete;

    std::size_t global_size() const noexcept { return global_size_; }
    std::size_t local_size() const noexcept { return local_size_; }
    std::ptrdiff_t local_offset() const noexcept { return local_offset_; }
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return rank_; }

    // Host data not available for device placement
    T* host_data() noexcept { return nullptr; }
    const T* host_data() const noexcept { return nullptr; }

    // Device data
    T* device_data() noexcept { return device_ptr_; }
    const T* device_data() const noexcept { return device_ptr_; }

    dtl_status copy_to_host(void* host_buffer, std::size_t count) const {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (count > local_size_) count = local_size_;
        auto err = cudaMemcpy(host_buffer, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }

    dtl_status copy_from_host(const void* host_buffer, std::size_t count) {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (count > local_size_) count = local_size_;
        auto err = cudaMemcpy(device_ptr_, host_buffer, count * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }

    dtl_status resize(std::size_t new_global_size) {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        T* old_ptr = device_ptr_;
        std::size_t old_local = local_size_;

        global_size_ = new_global_size;
        compute_local_partition();

        device_ptr_ = nullptr;
        if (local_size_ > 0) {
            auto err = cudaMalloc(&device_ptr_, local_size_ * sizeof(T));
            if (err != cudaSuccess) {
                device_ptr_ = old_ptr;
                return DTL_ERROR_ALLOCATION_FAILED;
            }
            // Copy preserved elements
            std::size_t copy_count = std::min(old_local, local_size_);
            if (copy_count > 0 && old_ptr) {
                cudaMemcpy(device_ptr_, old_ptr, copy_count * sizeof(T), cudaMemcpyDeviceToDevice);
            }
            // Zero-fill any new elements
            if (local_size_ > copy_count) {
                cudaMemset(static_cast<char*>(static_cast<void*>(device_ptr_)) + copy_count * sizeof(T),
                           0, (local_size_ - copy_count) * sizeof(T));
            }
        }
        if (old_ptr) cudaFree(old_ptr);
        return DTL_SUCCESS;
    }

    dtl_status fill(const T& value) {
        if (local_size_ == 0) return DTL_SUCCESS;
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        // Fill via host staging buffer
        std::vector<T> staging(local_size_, value);
        auto err = cudaMemcpy(device_ptr_, staging.data(), local_size_ * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }

    T reduce_sum() const {
        if (local_size_ == 0) return T{};
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        T sum{};
        for (const auto& val : staging) sum = sum + val;
        return sum;
    }

    T reduce_min() const {
        if (local_size_ == 0) return T{};
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        return *std::min_element(staging.begin(), staging.end());
    }

    T reduce_max() const {
        if (local_size_ == 0) return T{};
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        return *std::max_element(staging.begin(), staging.end());
    }

    void sort_ascending() {
        if (local_size_ <= 1) return;
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        std::sort(staging.begin(), staging.end());
        cudaMemcpy(device_ptr_, staging.data(), local_size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    void sort_descending() {
        if (local_size_ <= 1) return;
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        std::sort(staging.begin(), staging.end(), std::greater<T>());
        cudaMemcpy(device_ptr_, staging.data(), local_size_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    dtl_rank_t owner(dtl_index_t global_idx) const noexcept {
        if (global_idx < 0 || static_cast<std::size_t>(global_idx) >= global_size_) {
            return DTL_NO_RANK;
        }
        switch (partition_) {
            case DTL_PARTITION_BLOCK:   return owner_block(global_idx);
            case DTL_PARTITION_CYCLIC:  return owner_cyclic(global_idx);
            case DTL_PARTITION_REPLICATED: return rank_;
            default: return DTL_NO_RANK;
        }
    }

    bool is_local(dtl_index_t global_idx) const noexcept {
        if (partition_ == DTL_PARTITION_REPLICATED)
            return global_idx >= 0 && static_cast<std::size_t>(global_idx) < global_size_;
        return owner(global_idx) == rank_;
    }

    dtl_index_t to_local(dtl_index_t global_idx) const noexcept {
        if (!is_local(global_idx)) return -1;
        switch (partition_) {
            case DTL_PARTITION_BLOCK:      return global_idx - local_offset_;
            case DTL_PARTITION_CYCLIC:     return global_idx / num_ranks_;
            case DTL_PARTITION_REPLICATED: return global_idx;
            default: return -1;
        }
    }

    dtl_index_t to_global(dtl_index_t local_idx) const noexcept {
        if (local_idx < 0 || static_cast<std::size_t>(local_idx) >= local_size_) return -1;
        switch (partition_) {
            case DTL_PARTITION_BLOCK:      return local_offset_ + local_idx;
            case DTL_PARTITION_CYCLIC:     return local_idx * num_ranks_ + rank_;
            case DTL_PARTITION_REPLICATED: return local_idx;
            default: return -1;
        }
    }

private:
    void compute_local_partition() {
        switch (partition_) {
            case DTL_PARTITION_BLOCK: {
                std::size_t base = global_size_ / num_ranks_;
                std::size_t remainder = global_size_ % num_ranks_;
                if (static_cast<std::size_t>(rank_) < remainder) {
                    local_size_ = base + 1;
                    local_offset_ = rank_ * (base + 1);
                } else {
                    local_size_ = base;
                    local_offset_ = remainder * (base + 1) + (rank_ - remainder) * base;
                }
                break;
            }
            case DTL_PARTITION_CYCLIC: {
                std::size_t base = global_size_ / num_ranks_;
                std::size_t remainder = global_size_ % num_ranks_;
                local_size_ = base + (static_cast<std::size_t>(rank_) < remainder ? 1 : 0);
                local_offset_ = rank_;
                break;
            }
            case DTL_PARTITION_REPLICATED:
                local_size_ = global_size_;
                local_offset_ = 0;
                break;
            default:
                local_size_ = 0;
                local_offset_ = 0;
                break;
        }
    }

    dtl_rank_t owner_block(dtl_index_t global_idx) const noexcept {
        std::size_t base = global_size_ / num_ranks_;
        std::size_t remainder = global_size_ % num_ranks_;
        std::size_t boundary = remainder * (base + 1);
        if (static_cast<std::size_t>(global_idx) < boundary)
            return static_cast<dtl_rank_t>(global_idx / (base + 1));
        return static_cast<dtl_rank_t>(remainder + (global_idx - boundary) / base);
    }

    dtl_rank_t owner_cyclic(dtl_index_t global_idx) const noexcept {
        return static_cast<dtl_rank_t>(global_idx % num_ranks_);
    }

    std::size_t global_size_;
    std::size_t local_size_ = 0;
    std::ptrdiff_t local_offset_ = 0;
    dtl_rank_t rank_;
    dtl_rank_t num_ranks_;
    dtl_partition_policy partition_;
    std::size_t block_size_;
    int device_id_;
    T* device_ptr_;
};

// ============================================================================
// CUDA Unified Memory Vector Implementation
// ============================================================================

/**
 * @brief Concrete vector implementation for CUDA unified memory placement
 * @tparam T Element type (must be trivially_copyable)
 *
 * Stores data in CUDA managed memory (cudaMallocManaged).
 * Host-accessible: local_data() returns valid managed pointer.
 * Device-accessible: device_data() returns same managed pointer.
 */
template <typename T>
class unified_vector_impl {
public:
    unified_vector_impl(std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
                        dtl_partition_policy partition, int device_id,
                        std::size_t block_size = 1)
        : global_size_(global_size)
        , rank_(rank)
        , num_ranks_(num_ranks)
        , partition_(partition)
        , block_size_(block_size)
        , device_id_(device_id)
        , managed_ptr_(nullptr) {

        compute_local_partition();

        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (local_size_ > 0) {
            cudaMallocManaged(&managed_ptr_, local_size_ * sizeof(T));
            std::memset(managed_ptr_, 0, local_size_ * sizeof(T));
        }
    }

    ~unified_vector_impl() {
        if (managed_ptr_) {
            if (device_id_ >= 0) cudaSetDevice(device_id_);
            cudaFree(managed_ptr_);
        }
    }

    unified_vector_impl(const unified_vector_impl&) = delete;
    unified_vector_impl& operator=(const unified_vector_impl&) = delete;

    std::size_t global_size() const noexcept { return global_size_; }
    std::size_t local_size() const noexcept { return local_size_; }
    std::ptrdiff_t local_offset() const noexcept { return local_offset_; }
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return rank_; }

    // Host-accessible (managed memory)
    T* data() noexcept { return managed_ptr_; }
    const T* data() const noexcept { return managed_ptr_; }

    // Device-accessible (same managed pointer)
    T* device_data() noexcept { return managed_ptr_; }
    const T* device_data() const noexcept { return managed_ptr_; }

    dtl_status copy_to_host(void* host_buffer, std::size_t count) const {
        if (count > local_size_) count = local_size_;
        // Managed memory is host-accessible, sync then memcpy
        if (device_id_ >= 0) cudaDeviceSynchronize();
        std::memcpy(host_buffer, managed_ptr_, count * sizeof(T));
        return DTL_SUCCESS;
    }

    dtl_status copy_from_host(const void* host_buffer, std::size_t count) {
        if (count > local_size_) count = local_size_;
        std::memcpy(managed_ptr_, host_buffer, count * sizeof(T));
        return DTL_SUCCESS;
    }

    dtl_status resize(std::size_t new_global_size) {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        T* old_ptr = managed_ptr_;
        std::size_t old_local = local_size_;

        global_size_ = new_global_size;
        compute_local_partition();

        managed_ptr_ = nullptr;
        if (local_size_ > 0) {
            auto err = cudaMallocManaged(&managed_ptr_, local_size_ * sizeof(T));
            if (err != cudaSuccess) {
                managed_ptr_ = old_ptr;
                return DTL_ERROR_ALLOCATION_FAILED;
            }
            std::size_t copy_count = std::min(old_local, local_size_);
            if (copy_count > 0 && old_ptr) {
                if (device_id_ >= 0) cudaDeviceSynchronize();
                std::memcpy(managed_ptr_, old_ptr, copy_count * sizeof(T));
            }
            if (local_size_ > copy_count) {
                std::memset(managed_ptr_ + copy_count, 0, (local_size_ - copy_count) * sizeof(T));
            }
        }
        if (old_ptr) cudaFree(old_ptr);
        return DTL_SUCCESS;
    }

    dtl_status fill(const T& value) {
        if (device_id_ >= 0) cudaDeviceSynchronize();
        std::fill(managed_ptr_, managed_ptr_ + local_size_, value);
        return DTL_SUCCESS;
    }

    T reduce_sum() const {
        if (local_size_ == 0) return T{};
        if (device_id_ >= 0) cudaDeviceSynchronize();
        T sum{};
        for (std::size_t i = 0; i < local_size_; ++i) sum = sum + managed_ptr_[i];
        return sum;
    }

    T reduce_min() const {
        if (local_size_ == 0) return T{};
        if (device_id_ >= 0) cudaDeviceSynchronize();
        return *std::min_element(managed_ptr_, managed_ptr_ + local_size_);
    }

    T reduce_max() const {
        if (local_size_ == 0) return T{};
        if (device_id_ >= 0) cudaDeviceSynchronize();
        return *std::max_element(managed_ptr_, managed_ptr_ + local_size_);
    }

    void sort_ascending() {
        if (local_size_ <= 1) return;
        if (device_id_ >= 0) cudaDeviceSynchronize();
        std::sort(managed_ptr_, managed_ptr_ + local_size_);
    }

    void sort_descending() {
        if (local_size_ <= 1) return;
        if (device_id_ >= 0) cudaDeviceSynchronize();
        std::sort(managed_ptr_, managed_ptr_ + local_size_, std::greater<T>());
    }

    dtl_rank_t owner(dtl_index_t global_idx) const noexcept {
        if (global_idx < 0 || static_cast<std::size_t>(global_idx) >= global_size_)
            return DTL_NO_RANK;
        switch (partition_) {
            case DTL_PARTITION_BLOCK:   return owner_block(global_idx);
            case DTL_PARTITION_CYCLIC:  return owner_cyclic(global_idx);
            case DTL_PARTITION_REPLICATED: return rank_;
            default: return DTL_NO_RANK;
        }
    }

    bool is_local(dtl_index_t global_idx) const noexcept {
        if (partition_ == DTL_PARTITION_REPLICATED)
            return global_idx >= 0 && static_cast<std::size_t>(global_idx) < global_size_;
        return owner(global_idx) == rank_;
    }

    dtl_index_t to_local(dtl_index_t global_idx) const noexcept {
        if (!is_local(global_idx)) return -1;
        switch (partition_) {
            case DTL_PARTITION_BLOCK:      return global_idx - local_offset_;
            case DTL_PARTITION_CYCLIC:     return global_idx / num_ranks_;
            case DTL_PARTITION_REPLICATED: return global_idx;
            default: return -1;
        }
    }

    dtl_index_t to_global(dtl_index_t local_idx) const noexcept {
        if (local_idx < 0 || static_cast<std::size_t>(local_idx) >= local_size_) return -1;
        switch (partition_) {
            case DTL_PARTITION_BLOCK:      return local_offset_ + local_idx;
            case DTL_PARTITION_CYCLIC:     return local_idx * num_ranks_ + rank_;
            case DTL_PARTITION_REPLICATED: return local_idx;
            default: return -1;
        }
    }

private:
    void compute_local_partition() {
        switch (partition_) {
            case DTL_PARTITION_BLOCK: {
                std::size_t base = global_size_ / num_ranks_;
                std::size_t remainder = global_size_ % num_ranks_;
                if (static_cast<std::size_t>(rank_) < remainder) {
                    local_size_ = base + 1;
                    local_offset_ = rank_ * (base + 1);
                } else {
                    local_size_ = base;
                    local_offset_ = remainder * (base + 1) + (rank_ - remainder) * base;
                }
                break;
            }
            case DTL_PARTITION_CYCLIC: {
                std::size_t base = global_size_ / num_ranks_;
                std::size_t remainder = global_size_ % num_ranks_;
                local_size_ = base + (static_cast<std::size_t>(rank_) < remainder ? 1 : 0);
                local_offset_ = rank_;
                break;
            }
            case DTL_PARTITION_REPLICATED:
                local_size_ = global_size_;
                local_offset_ = 0;
                break;
            default:
                local_size_ = 0;
                local_offset_ = 0;
                break;
        }
    }

    dtl_rank_t owner_block(dtl_index_t global_idx) const noexcept {
        std::size_t base = global_size_ / num_ranks_;
        std::size_t remainder = global_size_ % num_ranks_;
        std::size_t boundary = remainder * (base + 1);
        if (static_cast<std::size_t>(global_idx) < boundary)
            return static_cast<dtl_rank_t>(global_idx / (base + 1));
        return static_cast<dtl_rank_t>(remainder + (global_idx - boundary) / base);
    }

    dtl_rank_t owner_cyclic(dtl_index_t global_idx) const noexcept {
        return static_cast<dtl_rank_t>(global_idx % num_ranks_);
    }

    std::size_t global_size_;
    std::size_t local_size_ = 0;
    std::ptrdiff_t local_offset_ = 0;
    dtl_rank_t rank_;
    dtl_rank_t num_ranks_;
    dtl_partition_policy partition_;
    std::size_t block_size_;
    int device_id_;
    T* managed_ptr_;
};

// ============================================================================
// CUDA Device Vector Vtable Factory
// ============================================================================

template <typename T>
const vector_vtable* get_device_vector_vtable() noexcept {
    static const vector_vtable vtable = {
        [](void* impl) { delete static_cast<device_vector_impl<T>*>(impl); },
        [](const void* impl) -> std::size_t { return static_cast<const device_vector_impl<T>*>(impl)->global_size(); },
        [](const void* impl) -> std::size_t { return static_cast<const device_vector_impl<T>*>(impl)->local_size(); },
        [](const void* impl) -> std::ptrdiff_t { return static_cast<const device_vector_impl<T>*>(impl)->local_offset(); },
        [](void*) -> void* { return nullptr; },  // local_data_mut (not host-accessible)
        [](const void*) -> const void* { return nullptr; },  // local_data
        [](void* impl) -> void* { return static_cast<device_vector_impl<T>*>(impl)->device_data(); },
        [](const void* impl) -> const void* { return static_cast<const device_vector_impl<T>*>(impl)->device_data(); },
        [](const void* impl, void* buf, std::size_t n) -> dtl_status { return static_cast<const device_vector_impl<T>*>(impl)->copy_to_host(buf, n); },
        [](void* impl, const void* buf, std::size_t n) -> dtl_status { return static_cast<device_vector_impl<T>*>(impl)->copy_from_host(buf, n); },
        [](void* impl, std::size_t s) -> dtl_status { return static_cast<device_vector_impl<T>*>(impl)->resize(s); },
        [](void* impl, const void* val) -> dtl_status { return static_cast<device_vector_impl<T>*>(impl)->fill(*static_cast<const T*>(val)); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const device_vector_impl<T>*>(impl)->num_ranks(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const device_vector_impl<T>*>(impl)->rank(); },
        [](const void* impl, dtl_index_t i) -> dtl_rank_t { return static_cast<const device_vector_impl<T>*>(impl)->owner(i); },
        [](const void* impl, dtl_index_t i) -> int { return static_cast<const device_vector_impl<T>*>(impl)->is_local(i) ? 1 : 0; },
        [](const void* impl, dtl_index_t i) -> dtl_index_t { return static_cast<const device_vector_impl<T>*>(impl)->to_local(i); },
        [](const void* impl, dtl_index_t i) -> dtl_index_t { return static_cast<const device_vector_impl<T>*>(impl)->to_global(i); },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const device_vector_impl<T>*>(impl)->reduce_sum(); return DTL_SUCCESS; },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const device_vector_impl<T>*>(impl)->reduce_min(); return DTL_SUCCESS; },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const device_vector_impl<T>*>(impl)->reduce_max(); return DTL_SUCCESS; },
        [](void* impl) -> dtl_status { static_cast<device_vector_impl<T>*>(impl)->sort_ascending(); return DTL_SUCCESS; },
        [](void* impl) -> dtl_status { static_cast<device_vector_impl<T>*>(impl)->sort_descending(); return DTL_SUCCESS; }
    };
    return &vtable;
}

// ============================================================================
// CUDA Unified Memory Vector Vtable Factory
// ============================================================================

template <typename T>
const vector_vtable* get_unified_vector_vtable() noexcept {
    static const vector_vtable vtable = {
        [](void* impl) { delete static_cast<unified_vector_impl<T>*>(impl); },
        [](const void* impl) -> std::size_t { return static_cast<const unified_vector_impl<T>*>(impl)->global_size(); },
        [](const void* impl) -> std::size_t { return static_cast<const unified_vector_impl<T>*>(impl)->local_size(); },
        [](const void* impl) -> std::ptrdiff_t { return static_cast<const unified_vector_impl<T>*>(impl)->local_offset(); },
        [](void* impl) -> void* { return static_cast<unified_vector_impl<T>*>(impl)->data(); },
        [](const void* impl) -> const void* { return static_cast<const unified_vector_impl<T>*>(impl)->data(); },
        [](void* impl) -> void* { return static_cast<unified_vector_impl<T>*>(impl)->device_data(); },
        [](const void* impl) -> const void* { return static_cast<const unified_vector_impl<T>*>(impl)->device_data(); },
        [](const void* impl, void* buf, std::size_t n) -> dtl_status { return static_cast<const unified_vector_impl<T>*>(impl)->copy_to_host(buf, n); },
        [](void* impl, const void* buf, std::size_t n) -> dtl_status { return static_cast<unified_vector_impl<T>*>(impl)->copy_from_host(buf, n); },
        [](void* impl, std::size_t s) -> dtl_status { return static_cast<unified_vector_impl<T>*>(impl)->resize(s); },
        [](void* impl, const void* val) -> dtl_status { return static_cast<unified_vector_impl<T>*>(impl)->fill(*static_cast<const T*>(val)); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const unified_vector_impl<T>*>(impl)->num_ranks(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const unified_vector_impl<T>*>(impl)->rank(); },
        [](const void* impl, dtl_index_t i) -> dtl_rank_t { return static_cast<const unified_vector_impl<T>*>(impl)->owner(i); },
        [](const void* impl, dtl_index_t i) -> int { return static_cast<const unified_vector_impl<T>*>(impl)->is_local(i) ? 1 : 0; },
        [](const void* impl, dtl_index_t i) -> dtl_index_t { return static_cast<const unified_vector_impl<T>*>(impl)->to_local(i); },
        [](const void* impl, dtl_index_t i) -> dtl_index_t { return static_cast<const unified_vector_impl<T>*>(impl)->to_global(i); },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const unified_vector_impl<T>*>(impl)->reduce_sum(); return DTL_SUCCESS; },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const unified_vector_impl<T>*>(impl)->reduce_min(); return DTL_SUCCESS; },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const unified_vector_impl<T>*>(impl)->reduce_max(); return DTL_SUCCESS; },
        [](void* impl) -> dtl_status { static_cast<unified_vector_impl<T>*>(impl)->sort_ascending(); return DTL_SUCCESS; },
        [](void* impl) -> dtl_status { static_cast<unified_vector_impl<T>*>(impl)->sort_descending(); return DTL_SUCCESS; }
    };
    return &vtable;
}

// ============================================================================
// CUDA Dispatch Helpers
// ============================================================================

template <typename T>
inline dtl_status create_device_vector(
    std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const vector_vtable** out_vtable, void** out_impl) {
    *out_vtable = get_device_vector_vtable<T>();
    *out_impl = new device_vector_impl<T>(global_size, rank, num_ranks, opts.partition, opts.device_id, opts.block_size);
    return DTL_SUCCESS;
}

template <typename T>
inline dtl_status create_unified_vector(
    std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const vector_vtable** out_vtable, void** out_impl) {
    *out_vtable = get_unified_vector_vtable<T>();
    *out_impl = new unified_vector_impl<T>(global_size, rank, num_ranks, opts.partition, opts.device_id, opts.block_size);
    return DTL_SUCCESS;
}

inline dtl_status dispatch_create_device_vector(
    dtl_dtype dtype, std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const vector_vtable** out_vtable, void** out_impl) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    return create_device_vector<int8_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT16:   return create_device_vector<int16_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT32:   return create_device_vector<int32_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT64:   return create_device_vector<int64_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT8: case DTL_DTYPE_BYTE: case DTL_DTYPE_BOOL:
                                return create_device_vector<uint8_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT16:  return create_device_vector<uint16_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT32:  return create_device_vector<uint32_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT64:  return create_device_vector<uint64_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_FLOAT32: return create_device_vector<float>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_FLOAT64: return create_device_vector<double>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        default: return DTL_ERROR_INVALID_ARGUMENT;
    }
}

inline dtl_status dispatch_create_unified_vector(
    dtl_dtype dtype, std::size_t global_size, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const vector_vtable** out_vtable, void** out_impl) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    return create_unified_vector<int8_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT16:   return create_unified_vector<int16_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT32:   return create_unified_vector<int32_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_INT64:   return create_unified_vector<int64_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT8: case DTL_DTYPE_BYTE: case DTL_DTYPE_BOOL:
                                return create_unified_vector<uint8_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT16:  return create_unified_vector<uint16_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT32:  return create_unified_vector<uint32_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_UINT64:  return create_unified_vector<uint64_t>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_FLOAT32: return create_unified_vector<float>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        case DTL_DTYPE_FLOAT64: return create_unified_vector<double>(global_size, rank, num_ranks, opts, out_vtable, out_impl);
        default: return DTL_ERROR_INVALID_ARGUMENT;
    }
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Dispatch Function
// ============================================================================

/**
 * @brief Create a vector implementation based on options
 * @param dtype Data type
 * @param global_size Global number of elements
 * @param rank Current rank
 * @param num_ranks Total number of ranks
 * @param opts Normalized options
 * @param[out] out_vtable Pointer to receive vtable
 * @param[out] out_impl Pointer to receive implementation
 * @return DTL_SUCCESS or error code
 */
inline dtl_status dispatch_create_vector(
    dtl_dtype dtype,
    std::size_t global_size,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    const stored_options& opts,
    const vector_vtable** out_vtable,
    void** out_impl) {

    // Route non-host placements to CUDA implementations when available
    if (opts.placement == DTL_PLACEMENT_DEVICE_PREFERRED) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

#if DTL_ENABLE_CUDA
    if (opts.placement == DTL_PLACEMENT_DEVICE) {
        return dispatch_create_device_vector(dtype, global_size, rank, num_ranks, opts, out_vtable, out_impl);
    }
    if (opts.placement == DTL_PLACEMENT_UNIFIED) {
        return dispatch_create_unified_vector(dtype, global_size, rank, num_ranks, opts, out_vtable, out_impl);
    }
#else
    if (opts.placement == DTL_PLACEMENT_DEVICE ||
        opts.placement == DTL_PLACEMENT_UNIFIED) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
#endif

    // Dispatch based on dtype
    try {
        switch (dtype) {
            case DTL_DTYPE_INT8: {
                *out_vtable = get_host_vector_vtable<int8_t>();
                *out_impl = new host_vector_impl<int8_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT16: {
                *out_vtable = get_host_vector_vtable<int16_t>();
                *out_impl = new host_vector_impl<int16_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT32: {
                *out_vtable = get_host_vector_vtable<int32_t>();
                *out_impl = new host_vector_impl<int32_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_INT64: {
                *out_vtable = get_host_vector_vtable<int64_t>();
                *out_impl = new host_vector_impl<int64_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT8:
            case DTL_DTYPE_BYTE:
            case DTL_DTYPE_BOOL: {
                *out_vtable = get_host_vector_vtable<uint8_t>();
                *out_impl = new host_vector_impl<uint8_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT16: {
                *out_vtable = get_host_vector_vtable<uint16_t>();
                *out_impl = new host_vector_impl<uint16_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT32: {
                *out_vtable = get_host_vector_vtable<uint32_t>();
                *out_impl = new host_vector_impl<uint32_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_UINT64: {
                *out_vtable = get_host_vector_vtable<uint64_t>();
                *out_impl = new host_vector_impl<uint64_t>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_FLOAT32: {
                *out_vtable = get_host_vector_vtable<float>();
                *out_impl = new host_vector_impl<float>(global_size, rank, num_ranks, opts.partition, opts.block_size);
                break;
            }
            case DTL_DTYPE_FLOAT64: {
                *out_vtable = get_host_vector_vtable<double>();
                *out_impl = new host_vector_impl<double>(global_size, rank, num_ranks, opts.partition, opts.block_size);
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

#endif /* DTL_C_DETAIL_VECTOR_DISPATCH_HPP */
