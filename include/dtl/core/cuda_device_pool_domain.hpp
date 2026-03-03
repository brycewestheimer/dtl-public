// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file cuda_device_pool_domain.hpp
/// @brief Multi-device CUDA domain for per-rank multi-GPU usage
/// @details Provides a domain that manages multiple CUDA devices and streams
///          within a single context, enabling advanced multi-GPU per MPI rank workflows.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/error/result.hpp>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#include <dtl/cuda/device_guard.hpp>
#endif

#include <memory>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace dtl {

// ============================================================================
// CUDA Device Pool Domain Tag
// ============================================================================

/// @brief Tag type for CUDA device pool domain
struct cuda_device_pool_domain_tag {};

// ============================================================================
// CUDA Device Pool Domain
// ============================================================================

#if DTL_ENABLE_CUDA

/// @brief Multi-device CUDA execution domain
/// @details Manages multiple CUDA devices and streams in a single domain,
///          enabling efficient multi-GPU usage within one process/rank.
///          Each device in the pool has its own default stream.
///
/// @par Use Cases
/// - Single MPI rank driving multiple GPUs
/// - Data-parallel workloads split across devices
/// - Pipeline parallelism with device-to-device transfers
///
/// @par Comparison with Multiple Contexts
/// | Approach | Pros | Cons |
/// |----------|------|------|
/// | `cuda_device_pool_domain` | Single context, unified lifetime | Slightly more complex API |
/// | Multiple contexts | Simple per-device contexts | Multiple context objects to manage |
///
/// @par Example Usage
/// @code
/// // Create pool with specific devices
/// auto pool_result = cuda_device_pool_domain::create({0, 1});
/// if (!pool_result) { /* handle error */ }
/// auto pool = std::move(*pool_result);
///
/// // Get stream for each device
/// cudaStream_t stream0 = pool.stream(0);
/// cudaStream_t stream1 = pool.stream(1);
///
/// // Launch kernels on different devices
/// {
///     auto guard = pool.make_device_guard(0);
///     launch_kernel<<<..., stream0>>>(...);
/// }
/// {
///     auto guard = pool.make_device_guard(1);
///     launch_kernel<<<..., stream1>>>(...);
/// }
///
/// // Synchronize all devices
/// pool.synchronize_all();
/// @endcode
class cuda_device_pool_domain {
public:
    using tag_type = cuda_device_pool_domain_tag;

    /// @brief Device info for a single GPU in the pool
    struct device_info {
        int device_id;          ///< CUDA device ID
        cudaStream_t stream;    ///< Default stream for this device
        bool owns_stream;       ///< Whether we created (and should destroy) the stream
    };

    /// @brief Default constructor (empty pool)
    cuda_device_pool_domain() = default;

    /// @brief Move constructor
    cuda_device_pool_domain(cuda_device_pool_domain&& other) noexcept
        : devices_(std::move(other.devices_))
        , valid_(other.valid_) {
        other.valid_ = false;
    }

    /// @brief Move assignment
    cuda_device_pool_domain& operator=(cuda_device_pool_domain&& other) noexcept {
        if (this != &other) {
            cleanup();
            devices_ = std::move(other.devices_);
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    /// @brief Destructor cleans up streams
    ~cuda_device_pool_domain() {
        cleanup();
    }

    // Non-copyable
    cuda_device_pool_domain(const cuda_device_pool_domain&) = delete;
    cuda_device_pool_domain& operator=(const cuda_device_pool_domain&) = delete;

    /// @brief Factory: create pool from list of device IDs
    /// @param device_ids List of CUDA device IDs to include in pool
    /// @return Result containing the domain or error
    [[nodiscard]] static result<cuda_device_pool_domain> create(std::vector<int> device_ids) {
        if (device_ids.empty()) {
            return result<cuda_device_pool_domain>::failure(
                status{status_code::invalid_argument, no_rank,
                       "Device pool requires at least one device"});
        }

        // Validate and deduplicate device IDs
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return result<cuda_device_pool_domain>::failure(
                status{status_code::not_supported, no_rank,
                       "No CUDA devices available"});
        }

        // Remove duplicates and validate
        std::sort(device_ids.begin(), device_ids.end());
        device_ids.erase(std::unique(device_ids.begin(), device_ids.end()), device_ids.end());

        for (int id : device_ids) {
            if (id < 0 || id >= device_count) {
                return result<cuda_device_pool_domain>::failure(
                    status{status_code::invalid_argument, no_rank,
                           "Invalid device ID: " + std::to_string(id)});
            }
        }

        cuda_device_pool_domain domain;
        domain.devices_.reserve(device_ids.size());

        // Create streams for each device
        for (int device_id : device_ids) {
            cuda::device_guard guard(device_id);
            
            cudaStream_t stream;
            err = cudaStreamCreate(&stream);
            if (err != cudaSuccess) {
                // Cleanup already-created streams
                domain.cleanup();
                return result<cuda_device_pool_domain>::failure(
                    status{status_code::allocation_failed, no_rank,
                           "Failed to create stream for device " + std::to_string(device_id)});
            }

            domain.devices_.push_back({device_id, stream, true});
        }

        domain.valid_ = true;
        return result<cuda_device_pool_domain>::success(std::move(domain));
    }

    /// @brief Factory: create pool from all available devices
    /// @return Result containing the domain or error
    [[nodiscard]] static result<cuda_device_pool_domain> create_all() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            return result<cuda_device_pool_domain>::failure(
                status{status_code::not_supported, no_rank,
                       "No CUDA devices available"});
        }

        std::vector<int> device_ids;
        device_ids.reserve(static_cast<size_t>(device_count));
        for (int i = 0; i < device_count; ++i) {
            device_ids.push_back(i);
        }

        return create(std::move(device_ids));
    }

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept { return valid_; }

    /// @brief Get number of devices in pool
    [[nodiscard]] size_type device_count() const noexcept {
        return static_cast<size_type>(devices_.size());
    }

    /// @brief Get device IDs in the pool
    [[nodiscard]] std::vector<int> device_ids() const {
        std::vector<int> ids;
        ids.reserve(devices_.size());
        for (const auto& dev : devices_) {
            ids.push_back(dev.device_id);
        }
        return ids;
    }

    /// @brief Get stream for a specific device ID
    /// @param device_id CUDA device ID
    /// @return Stream for that device
    /// @throws std::out_of_range if device not in pool
    [[nodiscard]] cudaStream_t stream(int device_id) const {
        for (const auto& dev : devices_) {
            if (dev.device_id == device_id) {
                return dev.stream;
            }
        }
        throw std::out_of_range("Device " + std::to_string(device_id) + " not in pool");
    }

    /// @brief Get stream by pool index (0, 1, 2, ...)
    /// @param index Index into the device list
    /// @return Stream for that device
    /// @throws std::out_of_range if index out of bounds
    [[nodiscard]] cudaStream_t stream_at(size_type index) const {
        if (index >= devices_.size()) {
            throw std::out_of_range("Pool index " + std::to_string(index) + " out of range");
        }
        return devices_[index].stream;
    }

    /// @brief Get device ID by pool index
    /// @param index Index into the device list
    /// @return Device ID
    /// @throws std::out_of_range if index out of bounds
    [[nodiscard]] int device_id_at(size_type index) const {
        if (index >= devices_.size()) {
            throw std::out_of_range("Pool index " + std::to_string(index) + " out of range");
        }
        return devices_[index].device_id;
    }

    /// @brief Check if a device is in the pool
    /// @param device_id CUDA device ID
    /// @return true if device is in pool
    [[nodiscard]] bool contains(int device_id) const noexcept {
        for (const auto& dev : devices_) {
            if (dev.device_id == device_id) {
                return true;
            }
        }
        return false;
    }

    /// @brief Synchronize a specific device's stream
    /// @param device_id CUDA device ID
    void synchronize(int device_id) {
        cuda::device_guard guard(device_id);
        cudaStreamSynchronize(stream(device_id));
    }

    /// @brief Synchronize all devices in the pool
    void synchronize_all() {
        for (const auto& dev : devices_) {
            cuda::device_guard guard(dev.device_id);
            cudaStreamSynchronize(dev.stream);
        }
    }

    /// @brief Create a device guard for a specific device
    /// @param device_id CUDA device ID
    /// @return Device guard RAII object
    [[nodiscard]] cuda::device_guard make_device_guard(int device_id) const noexcept {
        return cuda::device_guard(device_id);
    }

    /// @brief Get the first device ID in the pool
    /// @return First device ID or -1 if empty
    [[nodiscard]] int primary_device_id() const noexcept {
        return devices_.empty() ? -1 : devices_[0].device_id;
    }

private:
    void cleanup() noexcept {
        for (auto& dev : devices_) {
            if (dev.owns_stream && dev.stream != nullptr) {
                cuda::device_guard guard(dev.device_id);
                cudaStreamDestroy(dev.stream);
                dev.stream = nullptr;
            }
        }
        devices_.clear();
        valid_ = false;
    }

    std::vector<device_info> devices_;
    bool valid_{false};
};

#else  // !DTL_ENABLE_CUDA

/// @brief CUDA device pool domain stub (when CUDA is disabled)
class cuda_device_pool_domain {
public:
    using tag_type = cuda_device_pool_domain_tag;

    cuda_device_pool_domain() = default;

    [[nodiscard]] static result<cuda_device_pool_domain> create(std::vector<int>) {
        return result<cuda_device_pool_domain>::failure(
            status{status_code::not_supported, no_rank, "CUDA not enabled"});
    }

    [[nodiscard]] static result<cuda_device_pool_domain> create_all() {
        return result<cuda_device_pool_domain>::failure(
            status{status_code::not_supported, no_rank, "CUDA not enabled"});
    }

    [[nodiscard]] bool valid() const noexcept { return false; }
    [[nodiscard]] size_type device_count() const noexcept { return 0; }
    [[nodiscard]] std::vector<int> device_ids() const { return {}; }
    [[nodiscard]] int primary_device_id() const noexcept { return -1; }
    [[nodiscard]] bool contains(int) const noexcept { return false; }
    void synchronize(int) {}
    void synchronize_all() {}
};

#endif  // DTL_ENABLE_CUDA

// ============================================================================
// Context Factory Extension
// ============================================================================

/// @brief Context with CUDA device pool domain
/// @details A context that includes a cuda_device_pool_domain for multi-GPU usage.
///          Use this when a single rank needs to drive multiple GPUs.
/// @since 0.1.0
template <typename BaseContext>
class context_with_device_pool {
public:
    /// @brief Construct from base context and device pool
    context_with_device_pool(BaseContext base, cuda_device_pool_domain pool)
        : base_(std::move(base))
        , pool_(std::move(pool)) {}

    /// @brief Get the base context
    [[nodiscard]] const BaseContext& base() const noexcept { return base_; }
    [[nodiscard]] BaseContext& base() noexcept { return base_; }

    /// @brief Get the device pool
    [[nodiscard]] const cuda_device_pool_domain& device_pool() const noexcept { return pool_; }
    [[nodiscard]] cuda_device_pool_domain& device_pool() noexcept { return pool_; }

    /// @brief Forward rank() to base context
    [[nodiscard]] rank_t rank() const noexcept { return base_.rank(); }

    /// @brief Forward size() to base context
    [[nodiscard]] rank_t size() const noexcept { return base_.size(); }

    /// @brief Check if pool and base are valid
    [[nodiscard]] bool valid() const noexcept { return base_.valid() && pool_.valid(); }

    /// @brief Get device ID for a specific pool index
    [[nodiscard]] int device_id_for(size_type index) const {
        return pool_.device_id_at(index);
    }

    /// @brief Get number of devices in pool
    [[nodiscard]] size_type device_count() const noexcept {
        return pool_.device_count();
    }

private:
    BaseContext base_;
    cuda_device_pool_domain pool_;
};

/// @brief Create a context with a device pool
/// @tparam Ctx Base context type
/// @param ctx Base context
/// @param device_ids Devices to include in pool
/// @return Result containing context with device pool
template <typename Ctx>
[[nodiscard]] result<context_with_device_pool<Ctx>> make_context_with_device_pool(
    Ctx ctx, std::vector<int> device_ids) {
    auto pool_result = cuda_device_pool_domain::create(std::move(device_ids));
    if (!pool_result) {
        return result<context_with_device_pool<Ctx>>::failure(pool_result.error());
    }
    return result<context_with_device_pool<Ctx>>::success(
        context_with_device_pool<Ctx>(std::move(ctx), std::move(*pool_result)));
}

}  // namespace dtl
