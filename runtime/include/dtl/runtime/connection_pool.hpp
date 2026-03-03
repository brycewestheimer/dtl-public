// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file connection_pool.hpp
/// @brief Communicator connection pooling scaffold
/// @details Provides the interface for pooling and reusing communicator
///          connections. Currently a scaffold with stub factory; the pool
///          implementations will be added per-backend as demand requires.
/// @since 0.1.0

#pragma once

#include <dtl/runtime/detail/runtime_export.hpp>
#include <dtl/error/result.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <utility>

namespace dtl::runtime {

// =============================================================================
// Pool Metrics
// =============================================================================

/// @brief Statistics for a connection pool
struct pool_metrics {
    uint64_t total_acquired{0};   ///< Cumulative acquire count
    uint64_t total_released{0};   ///< Cumulative release count
    uint32_t current_active{0};   ///< Currently checked-out handles
    uint32_t high_water_mark{0};  ///< Peak concurrent active handles
    uint32_t pool_size{0};        ///< Current idle connections in pool
    uint32_t pool_capacity{0};    ///< Maximum idle connections allowed
};

// =============================================================================
// Pool Handle (RAII guard)
// =============================================================================

/// @brief RAII handle for a pooled connection
/// @details Move-only. When destroyed, returns the connection to the pool
///          via the release callback. The held pointer is type-erased (void*);
///          use get<T>() for typed access.
class pool_handle {
public:
    using release_fn = std::function<void(void*)>;

    /// @brief Default constructor (invalid handle)
    pool_handle() = default;

    /// @brief Construct with a resource and release callback
    /// @param resource The pooled resource (type-erased)
    /// @param on_release Callback invoked on destruction to return resource
    pool_handle(void* resource, release_fn on_release)
        : resource_(resource), on_release_(std::move(on_release)) {}

    /// @brief Move constructor
    pool_handle(pool_handle&& other) noexcept
        : resource_(std::exchange(other.resource_, nullptr))
        , on_release_(std::move(other.on_release_)) {}

    /// @brief Move assignment
    pool_handle& operator=(pool_handle&& other) noexcept {
        if (this != &other) {
            release();
            resource_ = std::exchange(other.resource_, nullptr);
            on_release_ = std::move(other.on_release_);
        }
        return *this;
    }

    // Non-copyable
    pool_handle(const pool_handle&) = delete;
    pool_handle& operator=(const pool_handle&) = delete;

    /// @brief Destructor — releases the resource back to the pool
    ~pool_handle() { release(); }

    /// @brief Get typed access to the held resource
    /// @tparam T The expected resource type
    /// @return Pointer to the resource (caller must ensure correct type)
    template <typename T>
    T* get() const noexcept { return static_cast<T*>(resource_); }

    /// @brief Check if the handle holds a valid resource
    [[nodiscard]] bool valid() const noexcept { return resource_ != nullptr; }

    /// @brief Explicitly release the resource back to the pool
    void release() {
        if (resource_ && on_release_) {
            on_release_(resource_);
        }
        resource_ = nullptr;
        on_release_ = nullptr;
    }

private:
    void* resource_{nullptr};
    release_fn on_release_;
};

// =============================================================================
// Communicator Pool Interface
// =============================================================================

/// @brief Abstract interface for backend-specific communicator pools
/// @details Each backend can implement this interface to pool its communicator
///          objects (e.g., MPI_Comm clones, NCCL communicators).
class communicator_pool {
public:
    virtual ~communicator_pool() = default;

    /// @brief Acquire a communicator handle from the pool
    /// @return RAII handle wrapping the communicator, or error
    virtual dtl::result<pool_handle> acquire() = 0;

    /// @brief Get current pool metrics
    virtual pool_metrics metrics() const noexcept = 0;

    /// @brief Name of the backend this pool serves
    virtual std::string_view backend_name() const noexcept = 0;

    /// @brief Set the maximum idle connection capacity
    /// @param capacity New maximum number of idle connections
    virtual void set_capacity(uint32_t capacity) = 0;

    /// @brief Drain all idle connections from the pool
    virtual void drain() = 0;

    // Non-copyable, non-movable
    communicator_pool(const communicator_pool&) = delete;
    communicator_pool& operator=(const communicator_pool&) = delete;

protected:
    communicator_pool() = default;
};

// =============================================================================
// Factory
// =============================================================================

/// @brief Create a communicator pool for the named backend
/// @param backend Backend name (e.g., "mpi", "nccl")
/// @return Owning pointer to the pool, or error
/// @details The `"mpi"` backend duplicates `MPI_COMM_WORLD` when DTL is built
///          with MPI support and MPI has already been initialized. Unsupported
///          backends return status_code::not_supported. If MPI pooling is built
///          in but MPI is not initialized, returns status_code::invalid_state.
DTL_RUNTIME_API dtl::result<std::unique_ptr<communicator_pool>>
make_communicator_pool(std::string_view backend);

}  // namespace dtl::runtime
