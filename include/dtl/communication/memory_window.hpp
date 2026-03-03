// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file memory_window.hpp
/// @brief RAII memory window abstraction for RMA operations
/// @details Provides a safe, RAII-based memory window abstraction for
///          one-sided communication operations.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <utility>

namespace dtl {

// ============================================================================
// Memory Window Info
// ============================================================================

/// @brief Information about a memory window
struct window_info {
    /// @brief Base address of the window
    void* base = nullptr;

    /// @brief Size of the window in bytes
    size_type size = 0;

    /// @brief Whether the window allocated its own memory
    bool owns_memory = false;

    /// @brief Native handle to the underlying window
    void* native_handle = nullptr;
};

// ============================================================================
// Memory Window Implementation Interface
// ============================================================================

/// @brief Abstract interface for memory window implementations
/// @details Allows different backends (MPI, SHMEM, etc.) to provide
///          their own window implementations.
class memory_window_impl {
public:
    virtual ~memory_window_impl() = default;

    /// @brief Get the base address of the window
    [[nodiscard]] virtual void* base() const noexcept = 0;

    /// @brief Get the size of the window in bytes
    [[nodiscard]] virtual size_type size() const noexcept = 0;

    /// @brief Check if this window is valid
    [[nodiscard]] virtual bool valid() const noexcept = 0;

    /// @brief Get the native handle (implementation-defined)
    [[nodiscard]] virtual void* native_handle() const noexcept = 0;

    /// @brief Perform a fence synchronization
    [[nodiscard]] virtual result<void> fence(int assert_flags = 0) = 0;

    /// @brief Lock all windows (passive-target)
    [[nodiscard]] virtual result<void> lock_all() = 0;

    /// @brief Unlock all windows (passive-target)
    [[nodiscard]] virtual result<void> unlock_all() = 0;

    /// @brief Lock a specific target (passive-target)
    [[nodiscard]] virtual result<void> lock(rank_t target, rma_lock_mode mode) = 0;

    /// @brief Unlock a specific target (passive-target)
    [[nodiscard]] virtual result<void> unlock(rank_t target) = 0;

    /// @brief Flush operations to a specific target
    [[nodiscard]] virtual result<void> flush(rank_t target) = 0;

    /// @brief Flush operations to all targets
    [[nodiscard]] virtual result<void> flush_all() = 0;

    /// @brief Flush local completion for a specific target
    [[nodiscard]] virtual result<void> flush_local(rank_t target) = 0;

    /// @brief Flush local completion for all targets
    [[nodiscard]] virtual result<void> flush_local_all() = 0;

    // -------------------------------------------------------------------------
    // Data Transfer Operations
    // -------------------------------------------------------------------------

    /// @brief Put data to a remote window
    [[nodiscard]] virtual result<void> put(
        const void* origin, size_type size, rank_t target, size_type target_offset) = 0;

    /// @brief Get data from a remote window
    [[nodiscard]] virtual result<void> get(
        void* origin, size_type size, rank_t target, size_type target_offset) = 0;

    /// @brief Accumulate to a remote window
    [[nodiscard]] virtual result<void> accumulate(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_reduce_op op) = 0;

    /// @brief Fetch and operation
    [[nodiscard]] virtual result<void> fetch_and_op(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) = 0;

    /// @brief Compare and swap
    [[nodiscard]] virtual result<void> compare_and_swap(
        const void* origin, const void* compare, void* result_buf,
        size_type size, rank_t target, size_type target_offset) = 0;

    /// @brief Get-accumulate (combines get with accumulate)
    [[nodiscard]] virtual result<void> get_accumulate(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) = 0;

    // -------------------------------------------------------------------------
    // Non-Blocking RMA Operations
    // -------------------------------------------------------------------------

    /// @brief Opaque handle for async RMA requests
    /// @details Backends store implementation-specific request objects.
    struct rma_request_handle {
        void* handle = nullptr;      ///< Backend-specific handle
        bool completed = false;      ///< Whether operation has completed
    };

    /// @brief Non-blocking put data to a remote window
    /// @param origin Origin buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @param request Output request handle for completion tracking
    /// @return Result indicating whether the operation was successfully initiated
    [[nodiscard]] virtual result<void> async_put(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) {
        // Default: synchronous fallback
        auto res = put(origin, size, target, target_offset);
        request.completed = true;
        return res;
    }

    /// @brief Non-blocking get data from a remote window
    /// @param origin Buffer to receive data
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @param request Output request handle for completion tracking
    /// @return Result indicating whether the operation was successfully initiated
    [[nodiscard]] virtual result<void> async_get(
        void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) {
        // Default: synchronous fallback
        auto res = get(origin, size, target, target_offset);
        request.completed = true;
        return res;
    }

    /// @brief Test if an async operation has completed
    /// @param request The request handle to test
    /// @return true if completed, false if still pending, or error
    [[nodiscard]] virtual result<bool> test_async(rma_request_handle& request) {
        // Default: always complete (for synchronous fallback)
        return request.completed;
    }

    /// @brief Wait for an async operation to complete
    /// @param request The request handle to wait on
    /// @return Result indicating success or failure
    [[nodiscard]] virtual result<void> wait_async(rma_request_handle& request) {
        // Default: no-op (for synchronous fallback)
        request.completed = true;
        return result<void>{};
    }
};

// ============================================================================
// Null Window Implementation
// ============================================================================

/// @brief Null window implementation for testing and single-process use
class null_window_impl final : public memory_window_impl {
public:
    null_window_impl(void* base, size_type size, bool owns_memory)
        : base_(base), size_(size), owns_memory_(owns_memory), valid_(base != nullptr) {}

    ~null_window_impl() override {
        if (owns_memory_ && base_) {
            ::operator delete(base_);
        }
    }

    [[nodiscard]] void* base() const noexcept override { return base_; }
    [[nodiscard]] size_type size() const noexcept override { return size_; }
    [[nodiscard]] bool valid() const noexcept override { return valid_; }
    [[nodiscard]] void* native_handle() const noexcept override { return nullptr; }

    [[nodiscard]] result<void> fence(int /*assert_flags*/) override {
        return result<void>{};
    }

    [[nodiscard]] result<void> lock_all() override {
        return result<void>{};
    }

    [[nodiscard]] result<void> unlock_all() override {
        return result<void>{};
    }

    [[nodiscard]] result<void> lock(rank_t /*target*/, rma_lock_mode /*mode*/) override {
        return result<void>{};
    }

    [[nodiscard]] result<void> unlock(rank_t /*target*/) override {
        return result<void>{};
    }

    [[nodiscard]] result<void> flush(rank_t /*target*/) override {
        return result<void>{};
    }

    [[nodiscard]] result<void> flush_all() override {
        return result<void>{};
    }

    [[nodiscard]] result<void> flush_local(rank_t /*target*/) override {
        return result<void>{};
    }

    [[nodiscard]] result<void> flush_local_all() override {
        return result<void>{};
    }

    [[nodiscard]] result<void> put(
        const void* origin, size_type size, rank_t target, size_type target_offset) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if (origin == nullptr) {
            return status_code::invalid_argument;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_range;
        }
        std::memcpy(static_cast<std::byte*>(base_) + target_offset, origin, size);
        return result<void>{};
    }

    [[nodiscard]] result<void> get(
        void* origin, size_type size, rank_t target, size_type target_offset) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if (origin == nullptr) {
            return status_code::invalid_argument;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_range;
        }
        std::memcpy(origin, static_cast<const std::byte*>(base_) + target_offset, size);
        return result<void>{};
    }

    [[nodiscard]] result<void> accumulate(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_reduce_op op) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if (origin == nullptr && size > 0) {
            return status_code::invalid_argument;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        // Perform local accumulate operation
        auto* dest = static_cast<std::byte*>(base_) + target_offset;
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

    [[nodiscard]] result<void> fetch_and_op(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if ((origin == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(base_) + target_offset;

        // Fetch old value
        std::memcpy(result_buf, dest, size);

        // Apply operation
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

    [[nodiscard]] result<void> compare_and_swap(
        const void* origin, const void* compare, void* result_buf,
        size_type size, rank_t target, size_type target_offset) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if ((origin == nullptr || compare == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(base_) + target_offset;

        // Fetch current value
        std::memcpy(result_buf, dest, size);

        // Compare and swap if equal
        if (std::memcmp(dest, compare, size) == 0) {
            std::memcpy(dest, origin, size);
        }

        return result<void>{};
    }

    [[nodiscard]] result<void> get_accumulate(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
        if (!valid_) {
            return status_code::invalid_state;
        }
        if ((origin == nullptr || result_buf == nullptr) && size > 0) {
            return status_code::invalid_argument;
        }
        if (target_offset + size > size_) {
            return status_code::out_of_bounds;
        }
        if (target != 0) {
            return status_code::invalid_rank;
        }

        auto* dest = static_cast<std::byte*>(base_) + target_offset;

        // Fetch old value
        std::memcpy(result_buf, dest, size);

        // Apply operation
        apply_reduce_op(dest, origin, size, op);

        return result<void>{};
    }

private:
    /// @brief Apply a reduction operation element-by-element
    void apply_reduce_op(void* dest, const void* origin, size_type size, rma_reduce_op op) {
        // For simplicity, treat as array of bytes and apply operation byte-by-byte
        // In real implementation, this would be type-aware
        auto* d = static_cast<std::byte*>(dest);
        const auto* o = static_cast<const std::byte*>(origin);

        switch (op) {
            case rma_reduce_op::replace:
                std::memcpy(dest, origin, size);
                break;
            case rma_reduce_op::no_op:
                // Do nothing
                break;
            case rma_reduce_op::sum:
                // For demonstration, treat as uint8_t array
                for (size_type i = 0; i < size; ++i) {
                    d[i] = static_cast<std::byte>(
                        static_cast<uint8_t>(d[i]) + static_cast<uint8_t>(o[i]));
                }
                break;
            case rma_reduce_op::prod:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = static_cast<std::byte>(
                        static_cast<uint8_t>(d[i]) * static_cast<uint8_t>(o[i]));
                }
                break;
            case rma_reduce_op::min:
                for (size_type i = 0; i < size; ++i) {
                    if (static_cast<uint8_t>(o[i]) < static_cast<uint8_t>(d[i])) {
                        d[i] = o[i];
                    }
                }
                break;
            case rma_reduce_op::max:
                for (size_type i = 0; i < size; ++i) {
                    if (static_cast<uint8_t>(o[i]) > static_cast<uint8_t>(d[i])) {
                        d[i] = o[i];
                    }
                }
                break;
            case rma_reduce_op::band:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] & o[i];
                }
                break;
            case rma_reduce_op::bor:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] | o[i];
                }
                break;
            case rma_reduce_op::bxor:
                for (size_type i = 0; i < size; ++i) {
                    d[i] = d[i] ^ o[i];
                }
                break;
        }
    }

    void* base_;
    size_type size_;
    bool owns_memory_;
    bool valid_;
};

// ============================================================================
// Memory Window Class
// ============================================================================

/// @brief RAII wrapper for RMA memory windows
/// @details Provides a safe abstraction over memory windows used for
///          one-sided communication. Supports both user-provided and
///          library-allocated memory.
///
/// @par Thread Safety
/// Memory windows are not thread-safe. Concurrent access to the same
/// window from multiple threads requires external synchronization.
///
/// @par Example Usage
/// @code
/// // Create window from existing memory
/// std::vector<int> data(1000);
/// auto win_result = memory_window::create(data.data(), data.size() * sizeof(int));
/// if (win_result) {
///     auto& win = *win_result;
///     // Use window for RMA operations...
/// }
///
/// // Create window with library-allocated memory
/// auto alloc_result = memory_window::allocate(1024);
/// if (alloc_result) {
///     auto& win = *alloc_result;
///     // Window owns the memory
/// }
/// @endcode
class memory_window {
public:
    // -------------------------------------------------------------------------
    // Factory Methods
    // -------------------------------------------------------------------------

    /// @brief Create a memory window from existing memory
    /// @param base Base address of the memory region
    /// @param size Size of the memory region in bytes
    /// @return Result containing the window or an error
    [[nodiscard]] static result<memory_window> create(void* base, size_type size) {
        if (base == nullptr && size > 0) {
            return status_code::invalid_argument;
        }
        auto impl = std::make_unique<null_window_impl>(base, size, false);
        return memory_window{std::move(impl)};
    }

    /// @brief Create a memory window with allocated memory
    /// @param size Size to allocate in bytes
    /// @return Result containing the window or an error
    [[nodiscard]] static result<memory_window> allocate(size_type size) {
        if (size == 0) {
            auto impl = std::make_unique<null_window_impl>(nullptr, 0, false);
            return memory_window{std::move(impl)};
        }
        void* ptr = ::operator new(size, std::nothrow);
        if (!ptr) {
            return status_code::allocation_failed;
        }
        auto impl = std::make_unique<null_window_impl>(ptr, size, true);
        return memory_window{std::move(impl)};
    }

    /// @brief Create a memory window from a span
    /// @tparam T Element type
    /// @param data Span over the data
    /// @return Result containing the window or an error
    /// @note For const spans, the const-ness is preserved at the API level
    ///       (put operations will not modify read-only windows).
    template <typename T>
    [[nodiscard]] static result<memory_window> from_span(std::span<T> data) {
        // Cast away const for storage; RMA implementations handle read-only
        // memory appropriately (get-only access patterns).
        return create(const_cast<std::remove_const_t<T>*>(data.data()), data.size_bytes());
    }

    /// @brief Create a memory window with a custom implementation
    /// @param impl The implementation to use
    /// @return The memory window
    [[nodiscard]] static memory_window from_impl(std::unique_ptr<memory_window_impl> impl) {
        return memory_window{std::move(impl)};
    }

    // -------------------------------------------------------------------------
    // Constructors and Assignment
    // -------------------------------------------------------------------------

    /// @brief Default constructor (invalid window)
    memory_window() : impl_(nullptr) {}

    /// @brief Destructor (releases resources)
    ~memory_window() = default;

    /// @brief Move constructor
    memory_window(memory_window&& other) noexcept : impl_(std::move(other.impl_)) {}

    /// @brief Move assignment
    memory_window& operator=(memory_window&& other) noexcept {
        if (this != &other) {
            impl_ = std::move(other.impl_);
        }
        return *this;
    }

    /// @brief Deleted copy constructor
    memory_window(const memory_window&) = delete;

    /// @brief Deleted copy assignment
    memory_window& operator=(const memory_window&) = delete;

    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------

    /// @brief Get the base address of the window
    /// @return Pointer to the base of the window, or nullptr if invalid
    [[nodiscard]] void* base() const noexcept {
        return impl_ ? impl_->base() : nullptr;
    }

    /// @brief Get the size of the window in bytes
    /// @return Size in bytes, or 0 if invalid
    [[nodiscard]] size_type size() const noexcept {
        return impl_ ? impl_->size() : 0;
    }

    /// @brief Check if this window is valid
    /// @return true if the window is valid and usable
    [[nodiscard]] bool valid() const noexcept {
        return impl_ && impl_->valid();
    }

    /// @brief Get the native handle (implementation-defined)
    /// @return Native handle, or nullptr if invalid
    [[nodiscard]] void* native_handle() const noexcept {
        return impl_ ? impl_->native_handle() : nullptr;
    }

    /// @brief Get window information
    /// @return Window info struct
    [[nodiscard]] window_info info() const noexcept {
        if (!impl_) {
            return window_info{};
        }
        return window_info{
            impl_->base(),
            impl_->size(),
            false,  // We don't expose ownership info
            impl_->native_handle()
        };
    }

    /// @brief Bool conversion operator
    [[nodiscard]] explicit operator bool() const noexcept {
        return valid();
    }

    // -------------------------------------------------------------------------
    // Active-Target Synchronization
    // -------------------------------------------------------------------------

    /// @brief Perform a fence synchronization
    /// @param assert_flags Assertion flags (implementation-defined)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> fence(int assert_flags = 0) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->fence(assert_flags);
    }

    // -------------------------------------------------------------------------
    // Passive-Target Synchronization
    // -------------------------------------------------------------------------

    /// @brief Lock all windows for passive-target access
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> lock_all() {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->lock_all();
    }

    /// @brief Unlock all windows
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> unlock_all() {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->unlock_all();
    }

    /// @brief Lock a specific target for passive-target access
    /// @param target Target rank to lock
    /// @param mode Lock mode (exclusive or shared)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> lock(rank_t target, rma_lock_mode mode = rma_lock_mode::exclusive) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->lock(target, mode);
    }

    /// @brief Unlock a specific target
    /// @param target Target rank to unlock
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> unlock(rank_t target) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->unlock(target);
    }

    // -------------------------------------------------------------------------
    // Flush Operations
    // -------------------------------------------------------------------------

    /// @brief Flush operations to a specific target
    /// @param target Target rank to flush
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush(rank_t target) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->flush(target);
    }

    /// @brief Flush operations to all targets
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_all() {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->flush_all();
    }

    /// @brief Flush local completion for a specific target
    /// @param target Target rank
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_local(rank_t target) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->flush_local(target);
    }

    /// @brief Flush local completion for all targets
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_local_all() {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->flush_local_all();
    }

    // -------------------------------------------------------------------------
    // Data Transfer Operations
    // -------------------------------------------------------------------------

    /// @brief Put data to a remote window
    /// @param origin Local data to send
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> put(const void* origin, size_type size, rank_t target, size_type target_offset) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->put(origin, size, target, target_offset);
    }

    /// @brief Get data from a remote window
    /// @param origin Local buffer to receive data
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> get(void* origin, size_type size, rank_t target, size_type target_offset) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->get(origin, size, target, target_offset);
    }

    /// @brief Accumulate to a remote window
    /// @param origin Local data to accumulate
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @param op Reduction operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> accumulate(const void* origin, size_type size, rank_t target,
                                           size_type target_offset, rma_reduce_op op) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->accumulate(origin, size, target, target_offset, op);
    }

    /// @brief Fetch and operation
    /// @param origin Local data for operation
    /// @param result_buf Buffer to receive old value
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @param op Reduction operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> fetch_and_op(const void* origin, void* result_buf, size_type size,
                                             rank_t target, size_type target_offset, rma_reduce_op op) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->fetch_and_op(origin, result_buf, size, target, target_offset, op);
    }

    /// @brief Compare and swap
    /// @param origin Value to swap in if compare matches
    /// @param compare Value to compare against
    /// @param result_buf Buffer to receive old value
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> compare_and_swap(const void* origin, const void* compare, void* result_buf,
                                                 size_type size, rank_t target, size_type target_offset) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->compare_and_swap(origin, compare, result_buf, size, target, target_offset);
    }

    /// @brief Get-accumulate (combines get with accumulate)
    /// @param origin Local data for operation
    /// @param result_buf Buffer to receive old value
    /// @param size Size of data in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window in bytes
    /// @param op Reduction operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> get_accumulate(const void* origin, void* result_buf, size_type size,
                                               rank_t target, size_type target_offset, rma_reduce_op op) {
        if (!impl_) {
            return status_code::invalid_state;
        }
        return impl_->get_accumulate(origin, result_buf, size, target, target_offset, op);
    }

    // -------------------------------------------------------------------------
    // Implementation Access (for backends)
    // -------------------------------------------------------------------------

    /// @brief Get the underlying implementation (for backends)
    /// @return Pointer to the implementation, or nullptr
    [[nodiscard]] memory_window_impl* get_impl() noexcept {
        return impl_.get();
    }

    /// @brief Get the underlying implementation (const, for backends)
    /// @return Const pointer to the implementation, or nullptr
    [[nodiscard]] const memory_window_impl* get_impl() const noexcept {
        return impl_.get();
    }

private:
    /// @brief Private constructor from implementation
    explicit memory_window(std::unique_ptr<memory_window_impl> impl)
        : impl_(std::move(impl)) {}

    std::unique_ptr<memory_window_impl> impl_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Check if two windows overlap
/// @param win1 First window
/// @param win2 Second window
/// @return true if the windows overlap in memory
[[nodiscard]] inline bool windows_overlap(const memory_window& win1,
                                          const memory_window& win2) noexcept {
    if (!win1.valid() || !win2.valid()) {
        return false;
    }

    auto base1 = static_cast<const std::byte*>(win1.base());
    auto end1 = base1 + win1.size();
    auto base2 = static_cast<const std::byte*>(win2.base());
    auto end2 = base2 + win2.size();

    return base1 < end2 && base2 < end1;
}

/// @brief Check if an offset and size are within window bounds
/// @param win The memory window
/// @param offset Offset in bytes
/// @param size Size in bytes
/// @return true if the range is within bounds
[[nodiscard]] inline bool window_range_valid(const memory_window& win,
                                              size_type offset,
                                              size_type size) noexcept {
    if (!win.valid()) {
        return false;
    }
    // Check for overflow and bounds
    return offset <= win.size() && size <= win.size() - offset;
}

}  // namespace dtl
