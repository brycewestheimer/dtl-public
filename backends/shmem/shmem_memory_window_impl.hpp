// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_memory_window_impl.hpp
/// @brief SHMEM-backed memory window implementation
/// @details Provides memory_window_impl using OpenSHMEM symmetric memory.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/memory_window.hpp>

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

#include <cstring>
#include <type_traits>

namespace dtl {
namespace shmem {

// ============================================================================
// SHMEM Memory Window Implementation
// ============================================================================

/// @brief SHMEM-backed memory window implementation
/// @details Implements DTL's memory_window_impl interface using OpenSHMEM
///          symmetric memory for PGAS-style one-sided communication.
///
/// @par Key Features:
/// - Symmetric memory allocation via shmem_malloc()
/// - One-sided put/get operations
/// - Fence/quiet synchronization
/// - Atomic operations (int, long types)
///
/// @par Memory Model:
/// SHMEM operations use symmetric memory - the same virtual address is
/// valid on all PEs. This window allocates symmetric memory and provides
/// RMA access through the DTL memory_window interface.
///
/// @par Synchronization:
/// - `fence()`: Orders operations before the fence with those after
/// - `flush_all()`: Waits for all operations to complete (via shmem_quiet)
/// - `lock()/unlock()`: Not required for SHMEM (passive-target by default)
///
/// @par Usage:
/// @code
/// // Create SHMEM window
/// auto win = std::make_unique<shmem_memory_window_impl>(1024);
///
/// // Put data to remote PE
/// win->put(local_data, sizeof(data), target_pe, offset);
/// win->flush_all();  // Ensure completion
///
/// // Atomics
/// win->fetch_and_op(&value, &result, sizeof(int), target_pe, offset, rma_reduce_op::sum);
/// @endcode
class shmem_memory_window_impl final : public memory_window_impl {
public:
    // -------------------------------------------------------------------------
    // Construction / Destruction
    // -------------------------------------------------------------------------

    /// @brief Construct window with specified size (allocates symmetric memory)
    /// @param size Size in bytes to allocate
    /// @note Collective operation - all PEs must call with same size
    explicit shmem_memory_window_impl(size_type size)
        : size_(size), owns_memory_(true) {
#if DTL_ENABLE_SHMEM
        base_ = shmem_malloc(size);
        valid_ = (base_ != nullptr);
        if (valid_) {
            // Zero-initialize for safety
            std::memset(base_, 0, size);
        }
        // Query PE info
        rank_ = shmem_my_pe();
        size_pes_ = shmem_n_pes();
#endif
    }

    /// @brief Construct window from existing symmetric memory
    /// @param base Pointer to existing symmetric memory
    /// @param size Size of the memory region
    /// @note Does not take ownership - caller must manage lifetime
    shmem_memory_window_impl(void* base, size_type size)
        : base_(base), size_(size), owns_memory_(false), valid_(base != nullptr) {
#if DTL_ENABLE_SHMEM
        rank_ = shmem_my_pe();
        size_pes_ = shmem_n_pes();
#endif
    }

    /// @brief Destructor - frees symmetric memory if owned
    ~shmem_memory_window_impl() override {
#if DTL_ENABLE_SHMEM
        if (owns_memory_ && base_ != nullptr) {
            shmem_free(base_);
        }
#endif
    }

    // Non-copyable
    shmem_memory_window_impl(const shmem_memory_window_impl&) = delete;
    shmem_memory_window_impl& operator=(const shmem_memory_window_impl&) = delete;

    // Movable
    shmem_memory_window_impl(shmem_memory_window_impl&& other) noexcept
        : base_(other.base_)
        , size_(other.size_)
        , owns_memory_(other.owns_memory_)
        , valid_(other.valid_)
        , rank_(other.rank_)
        , size_pes_(other.size_pes_) {
        other.base_ = nullptr;
        other.valid_ = false;
        other.owns_memory_ = false;
    }

    shmem_memory_window_impl& operator=(shmem_memory_window_impl&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_SHMEM
            if (owns_memory_ && base_ != nullptr) {
                shmem_free(base_);
            }
#endif
            base_ = other.base_;
            size_ = other.size_;
            owns_memory_ = other.owns_memory_;
            valid_ = other.valid_;
            rank_ = other.rank_;
            size_pes_ = other.size_pes_;

            other.base_ = nullptr;
            other.valid_ = false;
            other.owns_memory_ = false;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Basic Properties (memory_window_impl interface)
    // -------------------------------------------------------------------------

    [[nodiscard]] void* base() const noexcept override { return base_; }
    [[nodiscard]] size_type size() const noexcept override { return size_; }
    [[nodiscard]] bool valid() const noexcept override { return valid_; }
    [[nodiscard]] void* native_handle() const noexcept override { return base_; }

    // -------------------------------------------------------------------------
    // Synchronization (memory_window_impl interface)
    // -------------------------------------------------------------------------

    /// @brief Fence - orders operations before with those after
    [[nodiscard]] result<void> fence(int /*assert_flags*/ = 0) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        shmem_fence();
        return result<void>{};
#else
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Lock all - no-op for SHMEM (passive-target by default)
    [[nodiscard]] result<void> lock_all() override {
        // SHMEM doesn't require explicit locking
        return result<void>{};
    }

    /// @brief Unlock all - no-op for SHMEM
    [[nodiscard]] result<void> unlock_all() override {
        return result<void>{};
    }

    /// @brief Lock specific target - no-op for SHMEM
    [[nodiscard]] result<void> lock(rank_t /*target*/, rma_lock_mode /*mode*/) override {
        return result<void>{};
    }

    /// @brief Unlock specific target - no-op for SHMEM
    [[nodiscard]] result<void> unlock(rank_t /*target*/) override {
        return result<void>{};
    }

    /// @brief Flush to target - uses shmem_quiet for SHMEM
    [[nodiscard]] result<void> flush(rank_t /*target*/) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        // SHMEM doesn't have per-target flush, use quiet
        shmem_quiet();
        return result<void>{};
#else
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Flush all - ensures all operations complete via shmem_quiet
    [[nodiscard]] result<void> flush_all() override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        shmem_quiet();
        return result<void>{};
#else
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Flush local - same as flush for SHMEM
    [[nodiscard]] result<void> flush_local(rank_t /*target*/) override {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
        return result<void>{};
#else
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Flush local all - same as flush_all for SHMEM
    [[nodiscard]] result<void> flush_local_all() override {
        return flush_all();
    }

    // -------------------------------------------------------------------------
    // Data Transfer Operations
    // -------------------------------------------------------------------------

    /// @brief Put data to remote window
    [[nodiscard]] result<void> put(
        const void* origin, size_type size, rank_t target, size_type target_offset) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null origin buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }
        if (target_offset + size > size_) {
            return make_error<void>(status_code::out_of_range, "Offset + size exceeds window");
        }

        // Calculate target address (symmetric - same address on all PEs)
        auto* target_addr = static_cast<char*>(base_) + target_offset;
        shmem_putmem(target_addr, origin, size, target);
        return result<void>{};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Get data from remote window
    [[nodiscard]] result<void> get(
        void* origin, size_type size, rank_t target, size_type target_offset) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null origin buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }
        if (target_offset + size > size_) {
            return make_error<void>(status_code::out_of_range, "Offset + size exceeds window");
        }

        auto* source_addr = static_cast<const char*>(base_) + target_offset;
        shmem_getmem(origin, source_addr, size, target);
        return result<void>{};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Accumulate to remote window
    /// @note SHMEM supports limited atomic accumulate operations
    [[nodiscard]] result<void> accumulate(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_reduce_op op) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null origin buffer");
        }

        // SHMEM accumulate is only supported for specific types and operations
        // For general accumulate, we use fetch_and_op with a loop
        if (op == rma_reduce_op::replace) {
            // Replace is just a put
            return put(origin, size, target, target_offset);
        }

        if (op == rma_reduce_op::sum && size == sizeof(int)) {
            auto* target_addr = static_cast<int*>(base_) + (target_offset / sizeof(int));
            const int* src = static_cast<const int*>(origin);
            shmem_int_atomic_fetch_add(target_addr, *src, target);
            return result<void>{};
        }

        if (op == rma_reduce_op::sum && size == sizeof(long)) {
            auto* target_addr = static_cast<long*>(base_) + (target_offset / sizeof(long));
            const long* src = static_cast<const long*>(origin);
            shmem_long_atomic_fetch_add(target_addr, *src, target);
            return result<void>{};
        }

        return make_error<void>(status_code::not_supported,
                               "Accumulate operation not supported for this type/op");
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)op;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Fetch and operation
    [[nodiscard]] result<void> fetch_and_op(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr || result_buf == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }

        // Handle different types and operations
        if (size == sizeof(int)) {
            auto* target_addr = static_cast<int*>(base_) + (target_offset / sizeof(int));
            const int* src = static_cast<const int*>(origin);
            int* res = static_cast<int*>(result_buf);

            switch (op) {
                case rma_reduce_op::sum:
                    *res = shmem_int_atomic_fetch_add(target_addr, *src, target);
                    return result<void>{};
                case rma_reduce_op::replace:
                    *res = shmem_int_atomic_swap(target_addr, *src, target);
                    return result<void>{};
                case rma_reduce_op::no_op:
                    *res = shmem_int_atomic_fetch(target_addr, target);
                    return result<void>{};
                default:
                    return make_error<void>(status_code::not_supported,
                                           "Operation not supported for int");
            }
        }

        if (size == sizeof(long)) {
            auto* target_addr = static_cast<long*>(base_) + (target_offset / sizeof(long));
            const long* src = static_cast<const long*>(origin);
            long* res = static_cast<long*>(result_buf);

            switch (op) {
                case rma_reduce_op::sum:
                    *res = shmem_long_atomic_fetch_add(target_addr, *src, target);
                    return result<void>{};
                case rma_reduce_op::replace:
                    *res = shmem_long_atomic_swap(target_addr, *src, target);
                    return result<void>{};
                case rma_reduce_op::no_op:
                    *res = shmem_long_atomic_fetch(target_addr, target);
                    return result<void>{};
                default:
                    return make_error<void>(status_code::not_supported,
                                           "Operation not supported for long");
            }
        }

        return make_error<void>(status_code::not_supported,
                               "fetch_and_op not supported for this type");
#else
        (void)origin; (void)result_buf; (void)size; (void)target;
        (void)target_offset; (void)op;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Compare and swap
    [[nodiscard]] result<void> compare_and_swap(
        const void* origin, const void* compare, void* result_buf,
        size_type size, rank_t target, size_type target_offset) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr || compare == nullptr || result_buf == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }

        if (size == sizeof(int)) {
            auto* target_addr = static_cast<int*>(base_) + (target_offset / sizeof(int));
            const int* new_val = static_cast<const int*>(origin);
            const int* cmp_val = static_cast<const int*>(compare);
            int* res = static_cast<int*>(result_buf);

            *res = shmem_int_atomic_compare_swap(target_addr, *cmp_val, *new_val, target);
            return result<void>{};
        }

        if (size == sizeof(long)) {
            auto* target_addr = static_cast<long*>(base_) + (target_offset / sizeof(long));
            const long* new_val = static_cast<const long*>(origin);
            const long* cmp_val = static_cast<const long*>(compare);
            long* res = static_cast<long*>(result_buf);

            *res = shmem_long_atomic_compare_swap(target_addr, *cmp_val, *new_val, target);
            return result<void>{};
        }

        return make_error<void>(status_code::not_supported,
                               "compare_and_swap not supported for this type");
#else
        (void)origin; (void)compare; (void)result_buf; (void)size;
        (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Get-accumulate
    [[nodiscard]] result<void> get_accumulate(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
        // SHMEM get-accumulate can be implemented via fetch_and_op
        return fetch_and_op(origin, result_buf, size, target, target_offset, op);
    }

    // -------------------------------------------------------------------------
    // Non-Blocking Operations
    // -------------------------------------------------------------------------

    /// @brief Non-blocking put
    [[nodiscard]] result<void> async_put(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null origin buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }
        if (target_offset + size > size_) {
            return make_error<void>(status_code::out_of_range, "Offset + size exceeds window");
        }

        auto* target_addr = static_cast<char*>(base_) + target_offset;
        shmem_putmem_nbi(target_addr, origin, size, target);

        // SHMEM NBI operations complete on quiet(), not individually testable
        request.completed = false;
        request.handle = nullptr;
        return result<void>{};
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)request;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Non-blocking get
    [[nodiscard]] result<void> async_get(
        void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) override {
#if DTL_ENABLE_SHMEM
        if (!valid_) {
            return make_error<void>(status_code::invalid_state, "Window not valid");
        }
        if (origin == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null origin buffer");
        }
        if (target < 0 || target >= size_pes_) {
            return make_error<void>(status_code::invalid_rank, "Invalid target PE");
        }
        if (target_offset + size > size_) {
            return make_error<void>(status_code::out_of_range, "Offset + size exceeds window");
        }

        auto* source_addr = static_cast<const char*>(base_) + target_offset;
        shmem_getmem_nbi(origin, source_addr, size, target);

        request.completed = false;
        request.handle = nullptr;
        return result<void>{};
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)request;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    /// @brief Test async completion - SHMEM uses collective quiet
    [[nodiscard]] result<bool> test_async(rma_request_handle& request) override {
        // SHMEM doesn't have individual request testing
        // Operations complete collectively via quiet()
        return request.completed;
    }

    /// @brief Wait for async completion - calls shmem_quiet
    [[nodiscard]] result<void> wait_async(rma_request_handle& request) override {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
        request.completed = true;
        return result<void>{};
#else
        (void)request;
        return make_error<void>(status_code::not_supported, "SHMEM not enabled");
#endif
    }

    // -------------------------------------------------------------------------
    // SHMEM-Specific Methods
    // -------------------------------------------------------------------------

    /// @brief Get this PE's rank
    [[nodiscard]] rank_t rank() const noexcept { return rank_; }

    /// @brief Get total number of PEs
    [[nodiscard]] rank_t num_pes() const noexcept { return size_pes_; }

    /// @brief Barrier synchronization
    void barrier() {
#if DTL_ENABLE_SHMEM
        shmem_barrier_all();
#endif
    }

private:
    void* base_ = nullptr;
    size_type size_ = 0;
    bool owns_memory_ = false;
    bool valid_ = false;
    rank_t rank_ = 0;
    rank_t size_pes_ = 1;
};

// ============================================================================
// Factory Function
// ============================================================================

/// @brief Create a SHMEM memory window
/// @param size Size in bytes for the symmetric allocation
/// @return Result containing unique_ptr to the window or error
[[nodiscard]] inline result<std::unique_ptr<shmem_memory_window_impl>>
make_shmem_window(size_type size) {
#if DTL_ENABLE_SHMEM
    auto window = std::make_unique<shmem_memory_window_impl>(size);
    if (!window->valid()) {
        return make_error<std::unique_ptr<shmem_memory_window_impl>>(
            status_code::out_of_memory, "Failed to allocate SHMEM symmetric memory");
    }
    return window;
#else
    (void)size;
    return make_error<std::unique_ptr<shmem_memory_window_impl>>(
        status_code::not_supported, "SHMEM not enabled");
#endif
}

/// @brief Create a SHMEM memory window from existing symmetric memory
/// @param base Pointer to existing symmetric memory
/// @param size Size of the memory region
/// @return Result containing unique_ptr to the window or error
[[nodiscard]] inline result<std::unique_ptr<shmem_memory_window_impl>>
make_shmem_window(void* base, size_type size) {
#if DTL_ENABLE_SHMEM
    if (base == nullptr) {
        return make_error<std::unique_ptr<shmem_memory_window_impl>>(
            status_code::invalid_argument, "Null base pointer");
    }
    return std::make_unique<shmem_memory_window_impl>(base, size);
#else
    (void)base; (void)size;
    return make_error<std::unique_ptr<shmem_memory_window_impl>>(
        status_code::not_supported, "SHMEM not enabled");
#endif
}

}  // namespace shmem
}  // namespace dtl
