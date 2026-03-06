// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file tensor_dispatch.hpp
 * @brief Tensor creation dispatch for C ABI policy selection
 * @since 0.1.0
 *
 * This header implements create-time dispatch for tensors based on
 * runtime policy options.
 */

#ifndef DTL_C_DETAIL_TENSOR_DISPATCH_HPP
#define DTL_C_DETAIL_TENSOR_DISPATCH_HPP

#include "container_vtable.hpp"
#include "policy_matrix.hpp"
#include "placement_mapping.hpp"

#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_status.h>
#include <dtl/bindings/c/dtl_policies.h>

#include <algorithm>
#include <cstring>
#include <new>
#include <numeric>
#include <vector>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace dtl::c::detail {

// ============================================================================
// Host-Based Tensor Implementation
// ============================================================================

/**
 * @brief Concrete tensor implementation for host placement
 * @tparam T Element type
 *
 * Stores N-dimensional data in row-major (C) order with flat storage.
 * Distributes dimension 0 across ranks.
 */
template <typename T>
class host_tensor_impl {
public:
    host_tensor_impl(const dtl_shape& shape, dtl_rank_t rank, dtl_rank_t num_ranks)
        : global_shape_(shape)
        , ndim_(shape.ndim)
        , distributed_dim_(0)
        , my_rank_(rank)
        , num_ranks_(num_ranks) {

        // Compute local shape (partition dimension 0)
        local_shape_ = shape;
        partition_dim0();

        // Compute strides for local shape (row-major)
        compute_strides();

        // Compute total local size and allocate
        std::size_t total = 1;
        for (int i = 0; i < ndim_; ++i) {
            total *= local_shape_.dims[i];
        }
        data_.resize(total);
    }

    ~host_tensor_impl() = default;

    // Size queries
    std::size_t global_size() const noexcept {
        std::size_t total = 1;
        for (int i = 0; i < ndim_; ++i) {
            total *= global_shape_.dims[i];
        }
        return total;
    }

    std::size_t local_size() const noexcept { return data_.size(); }

    // Data access
    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    // Shape queries
    int ndim() const noexcept { return ndim_; }

    void shape(dtl_shape* out) const noexcept {
        *out = global_shape_;
    }

    void local_shape(dtl_shape* out) const noexcept {
        *out = local_shape_;
    }

    dtl_size_t dim(int d) const noexcept {
        if (d < 0 || d >= ndim_) return 0;
        return global_shape_.dims[d];
    }

    dtl_size_t stride(int d) const noexcept {
        if (d < 0 || d >= ndim_) return 0;
        return strides_[d];
    }

    int distributed_dim() const noexcept { return distributed_dim_; }

    // Distribution
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return my_rank_; }

    // ND index helpers
    std::size_t nd_to_linear(const dtl_index_t* indices) const noexcept {
        std::size_t idx = 0;
        for (int i = 0; i < ndim_; ++i) {
            idx += static_cast<std::size_t>(indices[i]) * strides_[i];
        }
        return idx;
    }

    bool nd_in_bounds(const dtl_index_t* indices) const noexcept {
        for (int i = 0; i < ndim_; ++i) {
            if (indices[i] < 0 || static_cast<dtl_size_t>(indices[i]) >= local_shape_.dims[i]) {
                return false;
            }
        }
        return true;
    }

    // Fill
    dtl_status fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
        return DTL_SUCCESS;
    }

    // Reduce sum
    T reduce_sum() const {
        T sum{};
        for (const auto& val : data_) {
            sum = sum + val;
        }
        return sum;
    }

    // Reshape
    dtl_status reshape(const dtl_shape* new_shape) {
        if (new_shape->ndim <= 0 || new_shape->ndim > DTL_MAX_TENSOR_RANK) {
            return DTL_ERROR_INVALID_ARGUMENT;
        }

        // Check that total global size matches
        std::size_t old_total = global_size();
        std::size_t new_total = 1;
        for (int i = 0; i < new_shape->ndim; ++i) {
            new_total *= new_shape->dims[i];
        }
        if (old_total != new_total) {
            return DTL_ERROR_DIMENSION_MISMATCH;
        }

        // Pre-validate: compute candidate local shape without modifying state
        dtl_size_t candidate_local_dim0;
        {
            dtl_size_t global_dim0 = new_shape->dims[0];
            dtl_size_t base = global_dim0 / num_ranks_;
            dtl_rank_t remainder = static_cast<dtl_rank_t>(global_dim0 % num_ranks_);
            candidate_local_dim0 = (my_rank_ < remainder) ? base + 1 : base;
        }
        std::size_t candidate_local_size = static_cast<std::size_t>(candidate_local_dim0);
        for (int i = 1; i < new_shape->ndim; ++i) {
            candidate_local_size *= new_shape->dims[i];
        }
        if (candidate_local_size != data_.size()) {
            return DTL_ERROR_NOT_IMPLEMENTED;
        }

        // Validation passed — now update member state
        global_shape_ = *new_shape;
        ndim_ = new_shape->ndim;

        local_shape_ = *new_shape;
        partition_dim0();

        compute_strides();

        return DTL_SUCCESS;
    }

private:
    void partition_dim0() {
        dtl_size_t global_dim = global_shape_.dims[0];
        dtl_size_t base = global_dim / num_ranks_;
        dtl_rank_t remainder = static_cast<dtl_rank_t>(global_dim % num_ranks_);

        if (my_rank_ < remainder) {
            local_shape_.dims[0] = base + 1;
        } else {
            local_shape_.dims[0] = base;
        }
    }

    void compute_strides() {
        if (ndim_ <= 0) return;
        strides_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * local_shape_.dims[i + 1];
        }
    }

    std::vector<T> data_;
    dtl_shape global_shape_;
    dtl_shape local_shape_;
    dtl_size_t strides_[DTL_MAX_TENSOR_RANK];
    int ndim_;
    int distributed_dim_;
    dtl_rank_t my_rank_;
    dtl_rank_t num_ranks_;
};

// ============================================================================
// Vtable Factory Functions
// ============================================================================

/**
 * @brief Create a vtable for a host tensor implementation
 */
template <typename T>
const tensor_vtable* get_host_tensor_vtable() noexcept {
    static const tensor_vtable vtable = {
        // destroy
        [](void* impl) {
            delete static_cast<host_tensor_impl<T>*>(impl);
        },

        // global_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->global_size();
        },

        // local_size
        [](const void* impl) -> std::size_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->local_size();
        },

        // local_data_mut
        [](void* impl) -> void* {
            return static_cast<host_tensor_impl<T>*>(impl)->data();
        },

        // local_data
        [](const void* impl) -> const void* {
            return static_cast<const host_tensor_impl<T>*>(impl)->data();
        },

        // device_data_mut
        [](void* /*impl*/) -> void* { return nullptr; },

        // device_data
        [](const void* /*impl*/) -> const void* { return nullptr; },

        // copy_to_host
        [](const void* impl, void* host_buffer, std::size_t count) -> dtl_status {
            const auto* self = static_cast<const host_tensor_impl<T>*>(impl);
            if (count > self->local_size()) count = self->local_size();
            std::memcpy(host_buffer, self->data(), count * sizeof(T));
            return DTL_SUCCESS;
        },

        // copy_from_host
        [](void* impl, const void* host_buffer, std::size_t count) -> dtl_status {
            auto* self = static_cast<host_tensor_impl<T>*>(impl);
            if (count > self->local_size()) count = self->local_size();
            std::memcpy(self->data(), host_buffer, count * sizeof(T));
            return DTL_SUCCESS;
        },

        // fill
        [](void* impl, const void* value) -> dtl_status {
            return static_cast<host_tensor_impl<T>*>(impl)->fill(*static_cast<const T*>(value));
        },

        // reduce_sum
        [](const void* impl, void* result) -> dtl_status {
            *static_cast<T*>(result) = static_cast<const host_tensor_impl<T>*>(impl)->reduce_sum();
            return DTL_SUCCESS;
        },

        // ndim
        [](const void* impl) -> int {
            return static_cast<const host_tensor_impl<T>*>(impl)->ndim();
        },

        // shape
        [](const void* impl, dtl_shape* out) {
            static_cast<const host_tensor_impl<T>*>(impl)->shape(out);
        },

        // local_shape
        [](const void* impl, dtl_shape* out) {
            static_cast<const host_tensor_impl<T>*>(impl)->local_shape(out);
        },

        // dim
        [](const void* impl, int dim) -> dtl_size_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->dim(dim);
        },

        // stride
        [](const void* impl, int dim) -> dtl_size_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->stride(dim);
        },

        // distributed_dim
        [](const void* impl) -> int {
            return static_cast<const host_tensor_impl<T>*>(impl)->distributed_dim();
        },

        // num_ranks
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->num_ranks();
        },

        // rank
        [](const void* impl) -> dtl_rank_t {
            return static_cast<const host_tensor_impl<T>*>(impl)->rank();
        },

        // get_local_nd
        [](const void* impl, const dtl_index_t* indices, void* value) -> dtl_status {
            auto* self = static_cast<const host_tensor_impl<T>*>(impl);
            if (!self->nd_in_bounds(indices)) return DTL_ERROR_OUT_OF_BOUNDS;
            std::size_t linear = self->nd_to_linear(indices);
            *static_cast<T*>(value) = self->data()[linear];
            return DTL_SUCCESS;
        },

        // set_local_nd
        [](void* impl, const dtl_index_t* indices, const void* value) -> dtl_status {
            auto* self = static_cast<host_tensor_impl<T>*>(impl);
            if (!self->nd_in_bounds(indices)) return DTL_ERROR_OUT_OF_BOUNDS;
            std::size_t linear = self->nd_to_linear(indices);
            self->data()[linear] = *static_cast<const T*>(value);
            return DTL_SUCCESS;
        },

        // get_local
        [](const void* impl, dtl_size_t linear_idx, void* value) -> dtl_status {
            auto* self = static_cast<const host_tensor_impl<T>*>(impl);
            if (linear_idx >= self->local_size()) return DTL_ERROR_OUT_OF_BOUNDS;
            *static_cast<T*>(value) = self->data()[linear_idx];
            return DTL_SUCCESS;
        },

        // set_local
        [](void* impl, dtl_size_t linear_idx, const void* value) -> dtl_status {
            auto* self = static_cast<host_tensor_impl<T>*>(impl);
            if (linear_idx >= self->local_size()) return DTL_ERROR_OUT_OF_BOUNDS;
            self->data()[linear_idx] = *static_cast<const T*>(value);
            return DTL_SUCCESS;
        },

        // reshape
        [](void* impl, const dtl_shape* new_shape) -> dtl_status {
            return static_cast<host_tensor_impl<T>*>(impl)->reshape(new_shape);
        }
    };
    return &vtable;
}

// ============================================================================
// CUDA Device/Unified Tensor Implementations
// ============================================================================

#if DTL_ENABLE_CUDA

/**
 * @brief Device tensor: data in CUDA device memory, ND access not supported
 */
template <typename T>
class device_tensor_impl {
public:
    device_tensor_impl(const dtl_shape& shape, dtl_rank_t rank, dtl_rank_t num_ranks, int device_id)
        : global_shape_(shape), ndim_(shape.ndim), my_rank_(rank), num_ranks_(num_ranks), device_id_(device_id), device_ptr_(nullptr) {
        local_shape_ = shape;
        partition_dim0();
        compute_strides();
        std::size_t total = compute_total();
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (total > 0) {
            cudaMalloc(&device_ptr_, total * sizeof(T));
            cudaMemset(device_ptr_, 0, total * sizeof(T));
        }
    }

    ~device_tensor_impl() {
        if (device_ptr_) {
            if (device_id_ >= 0) cudaSetDevice(device_id_);
            cudaFree(device_ptr_);
        }
    }

    std::size_t global_size() const noexcept {
        std::size_t total = 1;
        for (int i = 0; i < ndim_; ++i) total *= global_shape_.dims[i];
        return total;
    }
    std::size_t local_size() const noexcept { return local_size_; }
    T* device_data() noexcept { return device_ptr_; }
    const T* device_data() const noexcept { return device_ptr_; }
    int ndim() const noexcept { return ndim_; }
    void shape(dtl_shape* out) const noexcept { *out = global_shape_; }
    void local_shape(dtl_shape* out) const noexcept { *out = local_shape_; }
    dtl_size_t dim(int d) const noexcept { return (d >= 0 && d < ndim_) ? global_shape_.dims[d] : 0; }
    dtl_size_t stride(int d) const noexcept { return (d >= 0 && d < ndim_) ? strides_[d] : 0; }
    int distributed_dim() const noexcept { return 0; }
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return my_rank_; }

    dtl_status copy_to_host(void* buf, std::size_t count) const {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (count > local_size_) count = local_size_;
        auto err = cudaMemcpy(buf, device_ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }
    dtl_status copy_from_host(const void* buf, std::size_t count) {
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (count > local_size_) count = local_size_;
        auto err = cudaMemcpy(device_ptr_, buf, count * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }
    dtl_status fill(const T& value) {
        if (local_size_ == 0) return DTL_SUCCESS;
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        std::vector<T> staging(local_size_, value);
        auto err = cudaMemcpy(device_ptr_, staging.data(), local_size_ * sizeof(T), cudaMemcpyHostToDevice);
        return err == cudaSuccess ? DTL_SUCCESS : DTL_ERROR_UNKNOWN;
    }
    T reduce_sum() const {
        if (local_size_ == 0) return T{};
        std::vector<T> staging(local_size_);
        copy_to_host(staging.data(), local_size_);
        T sum{};
        for (const auto& v : staging) sum = sum + v;
        return sum;
    }

private:
    void partition_dim0() {
        dtl_size_t global_dim = global_shape_.dims[0];
        dtl_size_t base = global_dim / num_ranks_;
        dtl_rank_t remainder = static_cast<dtl_rank_t>(global_dim % num_ranks_);
        local_shape_.dims[0] = (my_rank_ < remainder) ? base + 1 : base;
    }
    void compute_strides() {
        if (ndim_ <= 0) return;
        strides_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i) strides_[i] = strides_[i + 1] * local_shape_.dims[i + 1];
    }
    std::size_t compute_total() {
        local_size_ = 1;
        for (int i = 0; i < ndim_; ++i) local_size_ *= local_shape_.dims[i];
        return local_size_;
    }

    dtl_shape global_shape_;
    dtl_shape local_shape_;
    dtl_size_t strides_[DTL_MAX_TENSOR_RANK]{};
    std::size_t local_size_ = 0;
    int ndim_;
    dtl_rank_t my_rank_;
    dtl_rank_t num_ranks_;
    int device_id_;
    T* device_ptr_;
};

/**
 * @brief Unified tensor: data in managed memory, ND access supported
 */
template <typename T>
class unified_tensor_impl {
public:
    unified_tensor_impl(const dtl_shape& shape, dtl_rank_t rank, dtl_rank_t num_ranks, int device_id)
        : global_shape_(shape), ndim_(shape.ndim), my_rank_(rank), num_ranks_(num_ranks), device_id_(device_id), managed_ptr_(nullptr) {
        local_shape_ = shape;
        partition_dim0();
        compute_strides();
        std::size_t total = compute_total();
        if (device_id_ >= 0) cudaSetDevice(device_id_);
        if (total > 0) {
            cudaMallocManaged(&managed_ptr_, total * sizeof(T));
            std::memset(managed_ptr_, 0, total * sizeof(T));
        }
    }

    ~unified_tensor_impl() {
        if (managed_ptr_) {
            if (device_id_ >= 0) cudaSetDevice(device_id_);
            cudaFree(managed_ptr_);
        }
    }

    std::size_t global_size() const noexcept {
        std::size_t total = 1;
        for (int i = 0; i < ndim_; ++i) total *= global_shape_.dims[i];
        return total;
    }
    std::size_t local_size() const noexcept { return local_size_; }
    T* data() noexcept { return managed_ptr_; }
    const T* data() const noexcept { return managed_ptr_; }
    T* device_data() noexcept { return managed_ptr_; }
    const T* device_data() const noexcept { return managed_ptr_; }
    int ndim() const noexcept { return ndim_; }
    void shape(dtl_shape* out) const noexcept { *out = global_shape_; }
    void local_shape(dtl_shape* out) const noexcept { *out = local_shape_; }
    dtl_size_t dim(int d) const noexcept { return (d >= 0 && d < ndim_) ? global_shape_.dims[d] : 0; }
    dtl_size_t stride(int d) const noexcept { return (d >= 0 && d < ndim_) ? strides_[d] : 0; }
    int distributed_dim() const noexcept { return 0; }
    dtl_rank_t num_ranks() const noexcept { return num_ranks_; }
    dtl_rank_t rank() const noexcept { return my_rank_; }

    dtl_status copy_to_host(void* buf, std::size_t count) const {
        if (count > local_size_) count = local_size_;
        if (device_id_ >= 0) cudaDeviceSynchronize();
        std::memcpy(buf, managed_ptr_, count * sizeof(T));
        return DTL_SUCCESS;
    }
    dtl_status copy_from_host(const void* buf, std::size_t count) {
        if (count > local_size_) count = local_size_;
        std::memcpy(managed_ptr_, buf, count * sizeof(T));
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

    std::size_t nd_to_linear(const dtl_index_t* indices) const noexcept {
        std::size_t idx = 0;
        for (int i = 0; i < ndim_; ++i) idx += static_cast<std::size_t>(indices[i]) * strides_[i];
        return idx;
    }
    bool nd_in_bounds(const dtl_index_t* indices) const noexcept {
        for (int i = 0; i < ndim_; ++i)
            if (indices[i] < 0 || static_cast<dtl_size_t>(indices[i]) >= local_shape_.dims[i]) return false;
        return true;
    }

    dtl_status reshape(const dtl_shape* new_shape) {
        if (new_shape->ndim <= 0 || new_shape->ndim > DTL_MAX_TENSOR_RANK) return DTL_ERROR_INVALID_ARGUMENT;
        std::size_t old_total = global_size();
        std::size_t new_total = 1;
        for (int i = 0; i < new_shape->ndim; ++i) new_total *= new_shape->dims[i];
        if (old_total != new_total) return DTL_ERROR_DIMENSION_MISMATCH;

        dtl_size_t candidate_local_dim0;
        {
            dtl_size_t global_dim0 = new_shape->dims[0];
            dtl_size_t base = global_dim0 / num_ranks_;
            dtl_rank_t remainder = static_cast<dtl_rank_t>(global_dim0 % num_ranks_);
            candidate_local_dim0 = (my_rank_ < remainder) ? base + 1 : base;
        }
        std::size_t candidate_local_size = static_cast<std::size_t>(candidate_local_dim0);
        for (int i = 1; i < new_shape->ndim; ++i) candidate_local_size *= new_shape->dims[i];
        if (candidate_local_size != local_size_) return DTL_ERROR_NOT_IMPLEMENTED;

        global_shape_ = *new_shape;
        ndim_ = new_shape->ndim;
        local_shape_ = *new_shape;
        partition_dim0();
        compute_strides();
        return DTL_SUCCESS;
    }

private:
    void partition_dim0() {
        dtl_size_t global_dim = global_shape_.dims[0];
        dtl_size_t base = global_dim / num_ranks_;
        dtl_rank_t remainder = static_cast<dtl_rank_t>(global_dim % num_ranks_);
        local_shape_.dims[0] = (my_rank_ < remainder) ? base + 1 : base;
    }
    void compute_strides() {
        if (ndim_ <= 0) return;
        strides_[ndim_ - 1] = 1;
        for (int i = ndim_ - 2; i >= 0; --i) strides_[i] = strides_[i + 1] * local_shape_.dims[i + 1];
    }
    std::size_t compute_total() {
        local_size_ = 1;
        for (int i = 0; i < ndim_; ++i) local_size_ *= local_shape_.dims[i];
        return local_size_;
    }

    dtl_shape global_shape_;
    dtl_shape local_shape_;
    dtl_size_t strides_[DTL_MAX_TENSOR_RANK]{};
    std::size_t local_size_ = 0;
    int ndim_;
    dtl_rank_t my_rank_;
    dtl_rank_t num_ranks_;
    int device_id_;
    T* managed_ptr_;
};

// ============================================================================
// CUDA Tensor Vtable Factories
// ============================================================================

template <typename T>
const tensor_vtable* get_device_tensor_vtable() noexcept {
    static const tensor_vtable vtable = {
        [](void* impl) { delete static_cast<device_tensor_impl<T>*>(impl); },
        [](const void* impl) -> std::size_t { return static_cast<const device_tensor_impl<T>*>(impl)->global_size(); },
        [](const void* impl) -> std::size_t { return static_cast<const device_tensor_impl<T>*>(impl)->local_size(); },
        [](void*) -> void* { return nullptr; },
        [](const void*) -> const void* { return nullptr; },
        [](void* impl) -> void* { return static_cast<device_tensor_impl<T>*>(impl)->device_data(); },
        [](const void* impl) -> const void* { return static_cast<const device_tensor_impl<T>*>(impl)->device_data(); },
        [](const void* impl, void* buf, std::size_t n) -> dtl_status { return static_cast<const device_tensor_impl<T>*>(impl)->copy_to_host(buf, n); },
        [](void* impl, const void* buf, std::size_t n) -> dtl_status { return static_cast<device_tensor_impl<T>*>(impl)->copy_from_host(buf, n); },
        [](void* impl, const void* val) -> dtl_status { return static_cast<device_tensor_impl<T>*>(impl)->fill(*static_cast<const T*>(val)); },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const device_tensor_impl<T>*>(impl)->reduce_sum(); return DTL_SUCCESS; },
        [](const void* impl) -> int { return static_cast<const device_tensor_impl<T>*>(impl)->ndim(); },
        [](const void* impl, dtl_shape* out) { static_cast<const device_tensor_impl<T>*>(impl)->shape(out); },
        [](const void* impl, dtl_shape* out) { static_cast<const device_tensor_impl<T>*>(impl)->local_shape(out); },
        [](const void* impl, int d) -> dtl_size_t { return static_cast<const device_tensor_impl<T>*>(impl)->dim(d); },
        [](const void* impl, int d) -> dtl_size_t { return static_cast<const device_tensor_impl<T>*>(impl)->stride(d); },
        [](const void* impl) -> int { return static_cast<const device_tensor_impl<T>*>(impl)->distributed_dim(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const device_tensor_impl<T>*>(impl)->num_ranks(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const device_tensor_impl<T>*>(impl)->rank(); },
        [](const void*, const dtl_index_t*, void*) -> dtl_status { return DTL_ERROR_NOT_SUPPORTED; },  // get_local_nd: device not host-dereferenceable
        [](void*, const dtl_index_t*, const void*) -> dtl_status { return DTL_ERROR_NOT_SUPPORTED; },  // set_local_nd
        [](const void*, dtl_size_t, void*) -> dtl_status { return DTL_ERROR_NOT_SUPPORTED; },           // get_local
        [](void*, dtl_size_t, const void*) -> dtl_status { return DTL_ERROR_NOT_SUPPORTED; },           // set_local
        [](void*, const dtl_shape*) -> dtl_status { return DTL_ERROR_NOT_SUPPORTED; }                   // reshape: not supported on device
    };
    return &vtable;
}

template <typename T>
const tensor_vtable* get_unified_tensor_vtable() noexcept {
    static const tensor_vtable vtable = {
        [](void* impl) { delete static_cast<unified_tensor_impl<T>*>(impl); },
        [](const void* impl) -> std::size_t { return static_cast<const unified_tensor_impl<T>*>(impl)->global_size(); },
        [](const void* impl) -> std::size_t { return static_cast<const unified_tensor_impl<T>*>(impl)->local_size(); },
        [](void* impl) -> void* { return static_cast<unified_tensor_impl<T>*>(impl)->data(); },
        [](const void* impl) -> const void* { return static_cast<const unified_tensor_impl<T>*>(impl)->data(); },
        [](void* impl) -> void* { return static_cast<unified_tensor_impl<T>*>(impl)->device_data(); },
        [](const void* impl) -> const void* { return static_cast<const unified_tensor_impl<T>*>(impl)->device_data(); },
        [](const void* impl, void* buf, std::size_t n) -> dtl_status { return static_cast<const unified_tensor_impl<T>*>(impl)->copy_to_host(buf, n); },
        [](void* impl, const void* buf, std::size_t n) -> dtl_status { return static_cast<unified_tensor_impl<T>*>(impl)->copy_from_host(buf, n); },
        [](void* impl, const void* val) -> dtl_status { return static_cast<unified_tensor_impl<T>*>(impl)->fill(*static_cast<const T*>(val)); },
        [](const void* impl, void* r) -> dtl_status { *static_cast<T*>(r) = static_cast<const unified_tensor_impl<T>*>(impl)->reduce_sum(); return DTL_SUCCESS; },
        [](const void* impl) -> int { return static_cast<const unified_tensor_impl<T>*>(impl)->ndim(); },
        [](const void* impl, dtl_shape* out) { static_cast<const unified_tensor_impl<T>*>(impl)->shape(out); },
        [](const void* impl, dtl_shape* out) { static_cast<const unified_tensor_impl<T>*>(impl)->local_shape(out); },
        [](const void* impl, int d) -> dtl_size_t { return static_cast<const unified_tensor_impl<T>*>(impl)->dim(d); },
        [](const void* impl, int d) -> dtl_size_t { return static_cast<const unified_tensor_impl<T>*>(impl)->stride(d); },
        [](const void* impl) -> int { return static_cast<const unified_tensor_impl<T>*>(impl)->distributed_dim(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const unified_tensor_impl<T>*>(impl)->num_ranks(); },
        [](const void* impl) -> dtl_rank_t { return static_cast<const unified_tensor_impl<T>*>(impl)->rank(); },
        // ND access works for unified (host-accessible)
        [](const void* impl, const dtl_index_t* indices, void* value) -> dtl_status {
            auto* self = static_cast<const unified_tensor_impl<T>*>(impl);
            if (!self->nd_in_bounds(indices)) return DTL_ERROR_OUT_OF_BOUNDS;
            if (self->device_data()) cudaDeviceSynchronize();
            *static_cast<T*>(value) = self->data()[self->nd_to_linear(indices)];
            return DTL_SUCCESS;
        },
        [](void* impl, const dtl_index_t* indices, const void* value) -> dtl_status {
            auto* self = static_cast<unified_tensor_impl<T>*>(impl);
            if (!self->nd_in_bounds(indices)) return DTL_ERROR_OUT_OF_BOUNDS;
            self->data()[self->nd_to_linear(indices)] = *static_cast<const T*>(value);
            return DTL_SUCCESS;
        },
        [](const void* impl, dtl_size_t linear_idx, void* value) -> dtl_status {
            auto* self = static_cast<const unified_tensor_impl<T>*>(impl);
            if (linear_idx >= self->local_size()) return DTL_ERROR_OUT_OF_BOUNDS;
            if (self->device_data()) cudaDeviceSynchronize();
            *static_cast<T*>(value) = self->data()[linear_idx];
            return DTL_SUCCESS;
        },
        [](void* impl, dtl_size_t linear_idx, const void* value) -> dtl_status {
            auto* self = static_cast<unified_tensor_impl<T>*>(impl);
            if (linear_idx >= self->local_size()) return DTL_ERROR_OUT_OF_BOUNDS;
            self->data()[linear_idx] = *static_cast<const T*>(value);
            return DTL_SUCCESS;
        },
        [](void* impl, const dtl_shape* new_shape) -> dtl_status { return static_cast<unified_tensor_impl<T>*>(impl)->reshape(new_shape); }
    };
    return &vtable;
}

// ============================================================================
// CUDA Tensor Dispatch Helpers
// ============================================================================

inline dtl_status dispatch_create_device_tensor(
    dtl_dtype dtype, const dtl_shape& shape, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const tensor_vtable** out_vtable, void** out_impl) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    { *out_vtable = get_device_tensor_vtable<int8_t>(); *out_impl = new device_tensor_impl<int8_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT16:   { *out_vtable = get_device_tensor_vtable<int16_t>(); *out_impl = new device_tensor_impl<int16_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT32:   { *out_vtable = get_device_tensor_vtable<int32_t>(); *out_impl = new device_tensor_impl<int32_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT64:   { *out_vtable = get_device_tensor_vtable<int64_t>(); *out_impl = new device_tensor_impl<int64_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT8: case DTL_DTYPE_BYTE: case DTL_DTYPE_BOOL:
                                { *out_vtable = get_device_tensor_vtable<uint8_t>(); *out_impl = new device_tensor_impl<uint8_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT16:  { *out_vtable = get_device_tensor_vtable<uint16_t>(); *out_impl = new device_tensor_impl<uint16_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT32:  { *out_vtable = get_device_tensor_vtable<uint32_t>(); *out_impl = new device_tensor_impl<uint32_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT64:  { *out_vtable = get_device_tensor_vtable<uint64_t>(); *out_impl = new device_tensor_impl<uint64_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_FLOAT32: { *out_vtable = get_device_tensor_vtable<float>(); *out_impl = new device_tensor_impl<float>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_FLOAT64: { *out_vtable = get_device_tensor_vtable<double>(); *out_impl = new device_tensor_impl<double>(shape, rank, num_ranks, opts.device_id); break; }
        default: return DTL_ERROR_INVALID_ARGUMENT;
    }
    return DTL_SUCCESS;
}

inline dtl_status dispatch_create_unified_tensor(
    dtl_dtype dtype, const dtl_shape& shape, dtl_rank_t rank, dtl_rank_t num_ranks,
    const stored_options& opts, const tensor_vtable** out_vtable, void** out_impl) {
    switch (dtype) {
        case DTL_DTYPE_INT8:    { *out_vtable = get_unified_tensor_vtable<int8_t>(); *out_impl = new unified_tensor_impl<int8_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT16:   { *out_vtable = get_unified_tensor_vtable<int16_t>(); *out_impl = new unified_tensor_impl<int16_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT32:   { *out_vtable = get_unified_tensor_vtable<int32_t>(); *out_impl = new unified_tensor_impl<int32_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_INT64:   { *out_vtable = get_unified_tensor_vtable<int64_t>(); *out_impl = new unified_tensor_impl<int64_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT8: case DTL_DTYPE_BYTE: case DTL_DTYPE_BOOL:
                                { *out_vtable = get_unified_tensor_vtable<uint8_t>(); *out_impl = new unified_tensor_impl<uint8_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT16:  { *out_vtable = get_unified_tensor_vtable<uint16_t>(); *out_impl = new unified_tensor_impl<uint16_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT32:  { *out_vtable = get_unified_tensor_vtable<uint32_t>(); *out_impl = new unified_tensor_impl<uint32_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_UINT64:  { *out_vtable = get_unified_tensor_vtable<uint64_t>(); *out_impl = new unified_tensor_impl<uint64_t>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_FLOAT32: { *out_vtable = get_unified_tensor_vtable<float>(); *out_impl = new unified_tensor_impl<float>(shape, rank, num_ranks, opts.device_id); break; }
        case DTL_DTYPE_FLOAT64: { *out_vtable = get_unified_tensor_vtable<double>(); *out_impl = new unified_tensor_impl<double>(shape, rank, num_ranks, opts.device_id); break; }
        default: return DTL_ERROR_INVALID_ARGUMENT;
    }
    return DTL_SUCCESS;
}

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Dispatch Function
// ============================================================================

/**
 * @brief Create a tensor implementation based on options
 */
inline dtl_status dispatch_create_tensor(
    dtl_dtype dtype,
    const dtl_shape& shape,
    dtl_rank_t rank,
    dtl_rank_t num_ranks,
    const stored_options& opts,
    const tensor_vtable** out_vtable,
    void** out_impl) {

    if (opts.placement == DTL_PLACEMENT_DEVICE_PREFERRED) {
        return DTL_ERROR_NOT_SUPPORTED;
    }

#if DTL_ENABLE_CUDA
    if (opts.placement == DTL_PLACEMENT_DEVICE) {
        return dispatch_create_device_tensor(dtype, shape, rank, num_ranks, opts, out_vtable, out_impl);
    }
    if (opts.placement == DTL_PLACEMENT_UNIFIED) {
        return dispatch_create_unified_tensor(dtype, shape, rank, num_ranks, opts, out_vtable, out_impl);
    }
#else
    if (opts.placement == DTL_PLACEMENT_DEVICE ||
        opts.placement == DTL_PLACEMENT_UNIFIED) {
        return DTL_ERROR_NOT_SUPPORTED;
    }
#endif

    try {
        switch (dtype) {
            case DTL_DTYPE_INT8: {
                *out_vtable = get_host_tensor_vtable<int8_t>();
                *out_impl = new host_tensor_impl<int8_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_INT16: {
                *out_vtable = get_host_tensor_vtable<int16_t>();
                *out_impl = new host_tensor_impl<int16_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_INT32: {
                *out_vtable = get_host_tensor_vtable<int32_t>();
                *out_impl = new host_tensor_impl<int32_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_INT64: {
                *out_vtable = get_host_tensor_vtable<int64_t>();
                *out_impl = new host_tensor_impl<int64_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_UINT8:
            case DTL_DTYPE_BYTE:
            case DTL_DTYPE_BOOL: {
                *out_vtable = get_host_tensor_vtable<uint8_t>();
                *out_impl = new host_tensor_impl<uint8_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_UINT16: {
                *out_vtable = get_host_tensor_vtable<uint16_t>();
                *out_impl = new host_tensor_impl<uint16_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_UINT32: {
                *out_vtable = get_host_tensor_vtable<uint32_t>();
                *out_impl = new host_tensor_impl<uint32_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_UINT64: {
                *out_vtable = get_host_tensor_vtable<uint64_t>();
                *out_impl = new host_tensor_impl<uint64_t>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_FLOAT32: {
                *out_vtable = get_host_tensor_vtable<float>();
                *out_impl = new host_tensor_impl<float>(shape, rank, num_ranks);
                break;
            }
            case DTL_DTYPE_FLOAT64: {
                *out_vtable = get_host_tensor_vtable<double>();
                *out_impl = new host_tensor_impl<double>(shape, rank, num_ranks);
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

#endif /* DTL_C_DETAIL_TENSOR_DISPATCH_HPP */
