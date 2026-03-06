// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file remote_ref.hpp
/// @brief Explicit handle for remote element access
/// @details remote_ref is "syntactically loud" - no implicit conversions.
///          Phase 12B: Extended with RMA window support for direct remote memory access.
///          Phase 08: Added true async operations with progress engine integration.
/// @since 0.1.0
/// @note Updated in 1.4.0: True async operations (no synchronous fallback).

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/core/traits.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/futures/distributed_future.hpp>
#include <dtl/futures/progress.hpp>

#include <cstdint>

namespace dtl {

/// @brief Remote element access capability state for a remote_ref
enum class remote_access_capability : std::uint8_t {
    local_only = 0,
    remote_transport_unavailable = 1,
    remote_rma = 2,
};

/// @brief Explicit handle for accessing remote distributed elements
/// @tparam T Element type
/// @details remote_ref provides a handle for elements that may or may not
///          be stored locally. It deliberately has NO implicit conversions
///          to enforce explicit acknowledgment of potential communication.
///
/// @par Design Rationale:
/// DTL does not pretend distribution is transparent. Remote access is
/// "syntactically loud" to prevent accidental per-element remote traffic
/// in loops that could devastate performance.
///
/// @par Deleted Operations:
/// - No implicit conversion to T& or const T&
/// - No implicit conversion to T* or const T*
/// - No implicit conversion to bool
/// - No operator* or operator->
///
/// @par Required Operations:
/// - get() - Explicit read (may communicate)
/// - put(v) - Explicit write (may communicate)
/// - is_local() - Check if communication will occur
/// - has_window() - Check if RMA window is available for remote access
///
/// @par Usage:
/// @code
/// distributed_vector<int> vec(1000, ctx);
/// auto ref = vec.global_view()[500];  // Returns remote_ref<int>
///
/// // WRONG - won't compile:
/// // int x = ref;              // No implicit conversion
/// // *ref = 42;                // No operator*
/// // if (ref) { ... }          // No bool conversion
///
/// // CORRECT:
/// if (ref.is_local()) {
///     int x = ref.get().value();  // Explicit, no communication
///     ref.put(42);                 // Explicit write
/// } else {
///     // Explicit acknowledgment of network operation
///     auto result = ref.async_get();
///     // ... do other work ...
///     int x = result.get().value();
/// }
/// @endcode
///
/// @warning This type is intentionally inconvenient to use. If you find
///          yourself using remote_ref in a loop, consider using local_view()
///          or segmented_view() instead for better performance.
template <typename T>
class remote_ref {
public:
    /// @brief The value type
    using value_type = std::remove_const_t<T>;

    /// @brief The element type (may be const)
    using element_type = T;

    // ========================================================================
    // EXPLICITLY DELETED IMPLICIT CONVERSIONS
    // ========================================================================

    /// @brief Deleted: No implicit conversion to T&
    operator T&() = delete;

    /// @brief Deleted: No implicit conversion to const T&
    operator const T&() const = delete;

    /// @brief Deleted: No implicit conversion to T*
    operator T*() = delete;

    /// @brief Deleted: No implicit conversion to const T*
    operator const T*() const = delete;

    /// @brief Deleted: No implicit bool conversion
    operator bool() const = delete;

    /// @brief Deleted: No dereference operator
    T& operator*() = delete;

    /// @brief Deleted: No const dereference operator
    const T& operator*() const = delete;

    /// @brief Deleted: No arrow operator
    T* operator->() = delete;

    /// @brief Deleted: No const arrow operator
    const T* operator->() const = delete;

    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /// @brief Construct a remote reference
    /// @param owner The rank that owns this element
    /// @param global_idx The global index of this element
    /// @param local_ptr Pointer to local storage (if local), nullptr otherwise
    /// @param remote_transport_available Whether remote transport is available
    remote_ref(rank_t owner, index_t global_idx, T* local_ptr = nullptr,
               bool remote_transport_available = true)
        : owner_{owner}
        , global_idx_{global_idx}
        , local_ptr_{local_ptr}
        , window_{nullptr}
        , remote_offset_{0}
        , remote_transport_available_{remote_transport_available} {}

    /// @brief Construct a remote reference with RMA window information
    /// @param owner The rank that owns this element
    /// @param global_idx The global index of this element
    /// @param local_ptr Pointer to local storage (if local), nullptr otherwise
    /// @param window Pointer to an RMA memory window for remote access (nullable)
    /// @param remote_offset Byte offset within the remote window for this element
    /// @param remote_transport_available Whether remote transport is available
    remote_ref(rank_t owner, index_t global_idx, T* local_ptr,
               memory_window_impl* window, size_type remote_offset,
               bool remote_transport_available = true)
        : owner_{owner}
        , global_idx_{global_idx}
        , local_ptr_{local_ptr}
        , window_{window}
        , remote_offset_{remote_offset}
        , remote_transport_available_{remote_transport_available} {}

    // ========================================================================
    // EXPLICIT ACCESS OPERATIONS
    // ========================================================================

    /// @brief Read value from element (may communicate)
    /// @return Result containing the value or an error
    /// @note For local elements with a valid local_ptr, returns the local value.
    ///       For remote elements with an RMA window, performs RMA get.
    ///       Otherwise returns not_supported.
    [[nodiscard]] result<value_type> get() const {
        if (is_local() && local_ptr_) {
            return result<value_type>::success(*local_ptr_);
        }
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            value_type value{};
            auto res = window_->get(&value, sizeof(T), owner_, remote_offset_);
            if (res.has_value()) {
                return result<value_type>::success(std::move(value));
            }
            return result<value_type>::failure(res.error());
        }
        // Remote access unavailable (transport disabled or no window)
        return result<value_type>::failure(
            status{status_code::not_supported});
    }

    /// @brief Write value to element (may communicate)
    /// @param value The value to write
    /// @return Result indicating success or error
    /// @note For local elements with a valid local_ptr, writes locally.
    ///       For remote elements with an RMA window, performs RMA put.
    ///       Otherwise returns not_supported.
    result<void> put(const value_type& value) {
        if (is_local() && local_ptr_) {
            *local_ptr_ = value;
            return result<void>{};
        }
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            return window_->put(&value, sizeof(T), owner_, remote_offset_);
        }
        // Remote access unavailable (transport disabled or no window)
        return result<void>{
            status{status_code::not_supported}};
    }

    /// @brief Write value to element (move version)
    /// @param value The value to write (moved)
    /// @return Result indicating success or error
    result<void> put(value_type&& value) {
        if (is_local() && local_ptr_) {
            *local_ptr_ = std::move(value);
            return result<void>{};
        }
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            return window_->put(&value, sizeof(T), owner_, remote_offset_);
        }
        // Remote access unavailable (transport disabled or no window)
        return result<void>{
            status{status_code::not_supported}};
    }

    // ========================================================================
    // ASYNC ACCESS OPERATIONS
    // ========================================================================

    /// @brief Asynchronously read value from element
    /// @return A distributed_future that will hold the value when complete
    /// @note Returns immediately without blocking. Completion requires
    ///       calling dtl::futures::poll() or enabling background progress.
    ///       For local elements, the future is immediately ready.
    /// @see docs/rma/async_remote_ref.md for async contract details
    [[nodiscard]] futures::distributed_future<value_type> async_get() const {
        auto state = std::make_shared<futures::shared_state<value_type>>();

        // Fast path: local access
        if (is_local() && local_ptr_) {
            state->set_value(*local_ptr_);
            return futures::distributed_future<value_type>(state);
        }

        // Remote access with window
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            // Create async request that integrates with progress engine
            auto request = std::make_shared<memory_window_impl::rma_request_handle>();
            auto value_storage = std::make_shared<value_type>();

            auto res = window_->async_get(value_storage.get(), sizeof(T), owner_, remote_offset_, *request);
            if (res.has_error()) {
                state->set_error(res.error());
                return futures::distributed_future<value_type>(state);
            }

            // If already complete (synchronous fallback or local window), set value immediately
            if (request->completed) {
                state->set_value(std::move(*value_storage));
                return futures::distributed_future<value_type>(state);
            }

            // Register progress callback to poll for completion
            auto* win = window_;
            futures::progress_engine::instance().register_callback(
                [state, request, win, value_storage]() -> bool {
                    auto test_res = win->test_async(*request);
                    if (test_res.has_error()) {
                        state->set_error(test_res.error());
                        return false;  // Complete with error
                    }
                    if (test_res.value()) {
                        state->set_value(std::move(*value_storage));
                        return false;  // Complete successfully
                    }
                    return true;  // Still pending
                }
            );

            return futures::distributed_future<value_type>(state);
        }

        // Remote access unavailable - error
        state->set_error(status{status_code::not_supported});
        return futures::distributed_future<value_type>(state);
    }

    /// @brief Asynchronously write value to element
    /// @param value The value to write
    /// @return A distributed_future<void> that resolves when the write completes
    /// @note Returns immediately without blocking. Completion requires
    ///       calling dtl::futures::poll() or enabling background progress.
    ///       For local elements, the future is immediately ready.
    /// @see docs/rma/async_remote_ref.md for async contract details
    [[nodiscard]] futures::distributed_future<void> async_put(const value_type& value) {
        auto state = std::make_shared<futures::shared_state<void>>();

        // Fast path: local access
        if (is_local() && local_ptr_) {
            *local_ptr_ = value;
            state->set_value();
            return futures::distributed_future<void>(state);
        }

        // Remote access with window
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            // Create async request
            auto request = std::make_shared<memory_window_impl::rma_request_handle>();
            auto value_copy = std::make_shared<value_type>(value);

            auto res = window_->async_put(value_copy.get(), sizeof(T), owner_, remote_offset_, *request);
            if (res.has_error()) {
                state->set_error(res.error());
                return futures::distributed_future<void>(state);
            }

            // If already complete, set ready immediately
            if (request->completed) {
                state->set_value();
                return futures::distributed_future<void>(state);
            }

            // Register progress callback
            auto* win = window_;
            futures::progress_engine::instance().register_callback(
                [state, request, win, value_copy]() -> bool {
                    auto test_res = win->test_async(*request);
                    if (test_res.has_error()) {
                        state->set_error(test_res.error());
                        return false;  // Complete with error
                    }
                    if (test_res.value()) {
                        state->set_value();
                        return false;  // Complete successfully
                    }
                    return true;  // Still pending
                }
            );

            return futures::distributed_future<void>(state);
        }

        // Remote access unavailable - error
        state->set_error(status{status_code::not_supported});
        return futures::distributed_future<void>(state);
    }

    // ========================================================================
    // QUERY OPERATIONS
    // ========================================================================

    /// @brief Get the owning rank of this element
    [[nodiscard]] rank_t owner_rank() const noexcept {
        return owner_;
    }

    /// @brief Get the global index of this element
    [[nodiscard]] index_t global_index() const noexcept {
        return global_idx_;
    }

    /// @brief Check if this element is local (no communication needed)
    /// @return true if the element is on the current rank
    [[nodiscard]] bool is_local() const noexcept {
        // Check if the referenced element is owned by the current rank.
        // A non-null local pointer indicates the element resides locally.
        return local_ptr_ != nullptr;
    }

    /// @brief Check if this element is remote (communication needed)
    [[nodiscard]] bool is_remote() const noexcept {
        return !is_local();
    }

    /// @brief Check if an RMA memory window is available for remote access
    /// @return true if a window pointer has been set
    [[nodiscard]] bool has_window() const noexcept {
        return window_ != nullptr;
    }

    /// @brief Check whether remote transport is available for this reference
    [[nodiscard]] bool remote_transport_available() const noexcept {
        return remote_transport_available_;
    }

    /// @brief Query remote access capability state
    [[nodiscard]] remote_access_capability remote_capability() const noexcept {
        if (is_local()) {
            return remote_access_capability::local_only;
        }
        if (remote_transport_available_ && window_ != nullptr) {
            return remote_access_capability::remote_rma;
        }
        return remote_access_capability::remote_transport_unavailable;
    }

private:
    rank_t owner_;                    ///< Rank that owns this element
    index_t global_idx_;              ///< Global index of this element
    T* local_ptr_;                    ///< Pointer to local storage (null if remote)
    memory_window_impl* window_;      ///< Nullable pointer to RMA memory window for remote access
    size_type remote_offset_;         ///< Byte offset within the remote window for this element
    bool remote_transport_available_; ///< Whether remote transport is available in current context
};

/// @brief Specialization for const types
template <typename T>
class remote_ref<const T> {
public:
    using value_type = T;
    using element_type = const T;

    // Same deleted operations as non-const version
    operator const T&() const = delete;
    operator const T*() const = delete;
    operator bool() const = delete;
    const T& operator*() const = delete;
    const T* operator->() const = delete;

    /// @brief Construct a const remote reference
    /// @param owner The rank that owns this element
    /// @param global_idx The global index of this element
    /// @param local_ptr Pointer to local storage (if local), nullptr otherwise
    /// @param remote_transport_available Whether remote transport is available
    remote_ref(rank_t owner, index_t global_idx, const T* local_ptr = nullptr,
               bool remote_transport_available = true)
        : owner_{owner}
        , global_idx_{global_idx}
        , local_ptr_{local_ptr}
        , window_{nullptr}
        , remote_offset_{0}
        , remote_transport_available_{remote_transport_available} {}

    /// @brief Construct a const remote reference with RMA window information
    /// @param owner The rank that owns this element
    /// @param global_idx The global index of this element
    /// @param local_ptr Pointer to local storage (if local), nullptr otherwise
    /// @param window Pointer to an RMA memory window for remote access (nullable)
    /// @param remote_offset Byte offset within the remote window for this element
    /// @param remote_transport_available Whether remote transport is available
    remote_ref(rank_t owner, index_t global_idx, const T* local_ptr,
               memory_window_impl* window, size_type remote_offset,
               bool remote_transport_available = true)
        : owner_{owner}
        , global_idx_{global_idx}
        , local_ptr_{local_ptr}
        , window_{window}
        , remote_offset_{remote_offset}
        , remote_transport_available_{remote_transport_available} {}

    /// @brief Read value (const version - only operation available)
    /// @return Result containing the value or an error
    /// @note For local elements with a valid local_ptr, returns the local value.
    ///       For remote elements with an RMA window, performs RMA get.
    ///       Otherwise returns not_supported.
    [[nodiscard]] result<T> get() const {
        if (is_local() && local_ptr_) {
            return result<T>::success(*local_ptr_);
        }
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            T value{};
            auto res = window_->get(&value, sizeof(T), owner_, remote_offset_);
            if (res.has_value()) {
                return result<T>::success(std::move(value));
            }
            return result<T>::failure(res.error());
        }
        return result<T>::failure(
            status{status_code::not_supported});
    }

    /// @brief Asynchronously read value from element (const version)
    /// @return A distributed_future that will hold the value when complete
    /// @note Returns immediately without blocking. Completion requires
    ///       calling dtl::futures::poll() or enabling background progress.
    [[nodiscard]] futures::distributed_future<T> async_get() const {
        auto state = std::make_shared<futures::shared_state<T>>();

        // Fast path: local access
        if (is_local() && local_ptr_) {
            state->set_value(*local_ptr_);
            return futures::distributed_future<T>(state);
        }

        // Remote access with window
        if (is_remote() && remote_capability() == remote_access_capability::remote_rma) {
            auto request = std::make_shared<memory_window_impl::rma_request_handle>();
            auto value_storage = std::make_shared<T>();

            auto res = window_->async_get(value_storage.get(), sizeof(T), owner_, remote_offset_, *request);
            if (res.has_error()) {
                state->set_error(res.error());
                return futures::distributed_future<T>(state);
            }

            if (request->completed) {
                state->set_value(std::move(*value_storage));
                return futures::distributed_future<T>(state);
            }

            auto* win = window_;
            futures::progress_engine::instance().register_callback(
                [state, request, win, value_storage]() -> bool {
                    auto test_res = win->test_async(*request);
                    if (test_res.has_error()) {
                        state->set_error(test_res.error());
                        return false;
                    }
                    if (test_res.value()) {
                        state->set_value(std::move(*value_storage));
                        return false;
                    }
                    return true;
                }
            );

            return futures::distributed_future<T>(state);
        }

        state->set_error(status{status_code::not_supported});
        return futures::distributed_future<T>(state);
    }

    /// @brief Get the owning rank of this element
    [[nodiscard]] rank_t owner_rank() const noexcept { return owner_; }

    /// @brief Get the global index of this element
    [[nodiscard]] index_t global_index() const noexcept { return global_idx_; }

    /// @brief Check if this element is local (no communication needed)
    [[nodiscard]] bool is_local() const noexcept { return local_ptr_ != nullptr; }

    /// @brief Check if this element is remote (communication needed)
    [[nodiscard]] bool is_remote() const noexcept { return !is_local(); }

    /// @brief Check if an RMA memory window is available for remote access
    [[nodiscard]] bool has_window() const noexcept { return window_ != nullptr; }

    /// @brief Check whether remote transport is available for this reference
    [[nodiscard]] bool remote_transport_available() const noexcept {
        return remote_transport_available_;
    }

    /// @brief Query remote access capability state
    [[nodiscard]] remote_access_capability remote_capability() const noexcept {
        if (is_local()) {
            return remote_access_capability::local_only;
        }
        if (remote_transport_available_ && window_ != nullptr) {
            return remote_access_capability::remote_rma;
        }
        return remote_access_capability::remote_transport_unavailable;
    }

private:
    rank_t owner_;                    ///< Rank that owns this element
    index_t global_idx_;              ///< Global index of this element
    const T* local_ptr_;              ///< Pointer to local storage (null if remote)
    memory_window_impl* window_;      ///< Nullable pointer to RMA memory window for remote access
    size_type remote_offset_;         ///< Byte offset within the remote window for this element
    bool remote_transport_available_; ///< Whether remote transport is available in current context
};

// =============================================================================
// Type Trait Specializations
// =============================================================================

/// @brief Specialization of is_remote_ref for remote_ref<T>
template <typename T>
struct is_remote_ref<remote_ref<T>> : std::true_type {};

}  // namespace dtl
