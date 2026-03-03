// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file remote_integration.hpp
/// @brief Integration of RMA with dtl::remote for efficient data movement
/// @details Provides rma_remote_ref for RMA-backed remote element access.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/communication/rma_operations.hpp>
#include <dtl/rma/async_rma.hpp>

namespace dtl::rma {

// ============================================================================
// RMA Remote Reference
// ============================================================================

/// @brief RMA-backed remote reference for efficient one-sided access
/// @tparam T Element type
/// @details Unlike message-passing based remote_ref, rma_remote_ref uses
///          one-sided RMA operations which can be more efficient for
///          fine-grained access patterns.
///
/// @par Design Notes:
/// - Uses memory_window for all operations
/// - get() performs RMA get (local copy from remote window)
/// - put() performs RMA put (local copy to remote window)
/// - No implicit conversions (by design)
///
/// @par Usage:
/// @code
/// memory_window window = ...;
/// rma_remote_ref<int> ref(1, 0, window);  // Element at rank 1, offset 0
///
/// // Read value
/// auto val = ref.get();
/// if (val) {
///     std::cout << "Got: " << *val << std::endl;
/// }
///
/// // Write value
/// auto result = ref.put(42);
/// @endcode
template <typename T>
class rma_remote_ref {
public:
    /// @brief Value type
    using value_type = std::remove_const_t<T>;

    /// @brief Element type (may be const)
    using element_type = T;

    // ========================================================================
    // EXPLICITLY DELETED IMPLICIT CONVERSIONS
    // ========================================================================

    operator T&() = delete;
    operator const T&() const = delete;
    operator T*() = delete;
    operator const T*() const = delete;
    operator bool() const = delete;
    T& operator*() = delete;
    const T& operator*() const = delete;
    T* operator->() = delete;
    const T* operator->() const = delete;

    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /// @brief Construct an RMA remote reference
    /// @param owner Rank that owns the target memory
    /// @param offset Byte offset in the target's window
    /// @param window Reference to the memory window
    /// @param local_rank The rank of this process (default 0 for single-process)
    rma_remote_ref(rank_t owner, size_type offset, memory_window& window,
                   rank_t local_rank = 0)
        : owner_(owner)
        , offset_(offset)
        , window_(&window)
        , local_rank_(local_rank) {}

    /// @brief Default constructor (invalid reference)
    rma_remote_ref() : owner_(0), offset_(0), window_(nullptr), local_rank_(0) {}

    // ========================================================================
    // EXPLICIT ACCESS OPERATIONS
    // ========================================================================

    /// @brief Read value using RMA get
    /// @return Result containing the value or an error
    [[nodiscard]] result<value_type> get() const {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }

        value_type value{};
        auto res = rma::get(owner_, offset_, value, *window_);
        if (!res.has_value()) {
            return res.error();
        }

        // Flush to ensure local completion
        auto flush_res = window_->flush_local(owner_);
        if (!flush_res.has_value()) {
            return flush_res.error();
        }

        return value;
    }

    /// @brief Write value using RMA put
    /// @param value The value to write
    /// @return Result indicating success or error
    result<void> put(const value_type& value) {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }

        auto res = rma::put(owner_, offset_, value, *window_);
        if (!res.has_value()) {
            return res.error();
        }

        // Flush to ensure remote completion
        return window_->flush(owner_);
    }

    /// @brief Write value using RMA put (move version)
    /// @param value The value to write
    /// @return Result indicating success or error
    result<void> put(value_type&& value) {
        return put(static_cast<const value_type&>(value));
    }

    // ========================================================================
    // ASYNC ACCESS OPERATIONS
    // ========================================================================

    /// @brief Asynchronously read value using RMA get
    /// @return Async get operation paired with its buffer
    /// @details Each call allocates a unique buffer to avoid data corruption
    ///          when multiple concurrent async gets are issued.
    [[nodiscard]] std::pair<std::unique_ptr<value_type>, async_get<value_type>> async_get_op() const {
        auto buffer = std::make_unique<value_type>();
        value_type* buf_ptr = buffer.get();
        auto op = async_get<value_type>(
            owner_, offset_, std::span<value_type>(buf_ptr, 1), *window_);
        return {std::move(buffer), std::move(op)};
    }

    /// @brief Asynchronously write value using RMA put
    /// @param value Span containing the value to write
    /// @return Async put operation
    [[nodiscard]] async_put<value_type> async_put_op(std::span<const value_type> value) const {
        return async_put<value_type>(owner_, offset_, value, *window_);
    }

    // ========================================================================
    // QUERY OPERATIONS
    // ========================================================================

    /// @brief Get the owning rank
    [[nodiscard]] rank_t owner_rank() const noexcept {
        return owner_;
    }

    /// @brief Get the byte offset in the window
    [[nodiscard]] size_type offset() const noexcept {
        return offset_;
    }

    /// @brief Check if this reference is valid
    [[nodiscard]] bool valid() const noexcept {
        return window_ != nullptr && window_->valid();
    }

    /// @brief Check if access is local (owner == self)
    [[nodiscard]] bool is_local() const noexcept {
        return owner_ == local_rank_;
    }

    /// @brief Check if access is remote
    [[nodiscard]] bool is_remote() const noexcept {
        return !is_local();
    }

    /// @brief Get the underlying window
    [[nodiscard]] memory_window* window() const noexcept {
        return window_;
    }

private:
    rank_t owner_;
    size_type offset_;
    memory_window* window_;
    rank_t local_rank_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create an RMA remote reference
/// @tparam T Element type
/// @param owner Owning rank
/// @param offset Byte offset in window
/// @param window Memory window
/// @param local_rank The rank of this process (default 0)
/// @return RMA remote reference
template <typename T>
[[nodiscard]] rma_remote_ref<T> make_rma_ref(rank_t owner, size_type offset,
                                              memory_window& window,
                                              rank_t local_rank = 0) {
    return rma_remote_ref<T>(owner, offset, window, local_rank);
}

/// @brief Create an RMA remote reference from element index
/// @tparam T Element type
/// @param owner Owning rank
/// @param index Element index (not byte offset)
/// @param window Memory window
/// @param local_rank The rank of this process (default 0)
/// @return RMA remote reference
template <typename T>
[[nodiscard]] rma_remote_ref<T> make_rma_ref_indexed(rank_t owner, size_type index,
                                                       memory_window& window,
                                                       rank_t local_rank = 0) {
    return rma_remote_ref<T>(owner, index * sizeof(T), window, local_rank);
}

// ============================================================================
// Const Specialization
// ============================================================================

/// @brief Specialization for const types (read-only access)
template <typename T>
class rma_remote_ref<const T> {
public:
    using value_type = T;
    using element_type = const T;

    // Deleted implicit conversions
    operator const T&() const = delete;
    operator const T*() const = delete;
    operator bool() const = delete;
    const T& operator*() const = delete;
    const T* operator->() const = delete;

    rma_remote_ref(rank_t owner, size_type offset, memory_window& window,
                   rank_t local_rank = 0)
        : owner_(owner), offset_(offset), window_(&window), local_rank_(local_rank) {}

    rma_remote_ref() : owner_(0), offset_(0), window_(nullptr), local_rank_(0) {}

    [[nodiscard]] result<T> get() const {
        if (!window_ || !window_->valid()) {
            return status_code::invalid_state;
        }

        T value{};
        auto res = rma::get(owner_, offset_, value, *window_);
        if (!res.has_value()) {
            return res.error();
        }

        auto flush_res = window_->flush_local(owner_);
        if (!flush_res.has_value()) {
            return flush_res.error();
        }

        return value;
    }

    [[nodiscard]] std::pair<std::unique_ptr<T>, async_get<T>> async_get_op() const {
        auto buffer = std::make_unique<T>();
        T* buf_ptr = buffer.get();
        auto op = async_get<T>(
            owner_, offset_, std::span<T>(buf_ptr, 1), *window_);
        return {std::move(buffer), std::move(op)};
    }

    [[nodiscard]] rank_t owner_rank() const noexcept { return owner_; }
    [[nodiscard]] size_type offset() const noexcept { return offset_; }
    [[nodiscard]] bool valid() const noexcept { return window_ && window_->valid(); }
    [[nodiscard]] bool is_local() const noexcept { return owner_ == local_rank_; }
    [[nodiscard]] bool is_remote() const noexcept { return !is_local(); }
    [[nodiscard]] memory_window* window() const noexcept { return window_; }

private:
    rank_t owner_;
    size_type offset_;
    memory_window* window_;
    rank_t local_rank_;
};

// ============================================================================
// Type Traits
// ============================================================================

/// @brief Type trait to check if a type is an rma_remote_ref
template <typename T>
struct is_rma_remote_ref : std::false_type {};

template <typename T>
struct is_rma_remote_ref<rma_remote_ref<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_rma_remote_ref_v = is_rma_remote_ref<T>::value;

}  // namespace dtl::rma
