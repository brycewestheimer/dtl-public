// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_window.hpp
/// @brief MPI RMA window implementation (Layer 1 - result<> returns)
/// @details Provides MPI-3 RMA window wrapper with RAII semantics.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>
#include <dtl/communication/memory_window.hpp>
#include <backends/mpi/mpi_communicator.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <utility>

namespace dtl {
namespace mpi {

// ============================================================================
// MPI RMA Window
// ============================================================================

/// @brief MPI-3 RMA window wrapper with RAII semantics
/// @details Wraps an MPI_Win and provides result-returning operations.
///          This is Layer 1 of the two-layer backend pattern.
class mpi_window {
public:
    using size_type = dtl::size_type;

    // ------------------------------------------------------------------------
    // Factory Methods
    // ------------------------------------------------------------------------

    /// @brief Create a window from existing memory
    /// @param base Pointer to local memory to expose
    /// @param size Size of memory region in bytes
    /// @param comm MPI communicator for window
    /// @return Created window or error
    [[nodiscard]] static result<mpi_window> create(
        void* base, size_type size, mpi_communicator& comm) {
#if DTL_ENABLE_MPI
        MPI_Win win;
        int mpi_result = MPI_Win_create(
            base,
            static_cast<MPI_Aint>(size),
            1,  // disp_unit
            MPI_INFO_NULL,
            comm.native_handle(),
            &win
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<mpi_window>(status_code::communication_error,
                                          "MPI_Win_create failed");
        }

        return mpi_window(win, base, size, false);
#else
        (void)base; (void)size; (void)comm;
        return make_error<mpi_window>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Allocate a window with new memory
    /// @param size Size of memory region in bytes
    /// @param comm MPI communicator for window
    /// @return Created window or error
    [[nodiscard]] static result<mpi_window> allocate(
        size_type size, mpi_communicator& comm) {
#if DTL_ENABLE_MPI
        MPI_Win win;
        void* base = nullptr;

        int mpi_result = MPI_Win_allocate(
            static_cast<MPI_Aint>(size),
            1,  // disp_unit
            MPI_INFO_NULL,
            comm.native_handle(),
            &base,
            &win
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<mpi_window>(status_code::allocation_failed,
                                          "MPI_Win_allocate failed");
        }

        return mpi_window(win, base, size, true);
#else
        (void)size; (void)comm;
        return make_error<mpi_window>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Constructors and Destructor
    // ------------------------------------------------------------------------

    /// @brief Default constructor (invalid window)
    mpi_window() = default;

    /// @brief Destructor - frees MPI window
    ~mpi_window() {
#if DTL_ENABLE_MPI
        if (win_ != MPI_WIN_NULL) {
            MPI_Win_free(&win_);
        }
#endif
    }

    // Non-copyable
    mpi_window(const mpi_window&) = delete;
    mpi_window& operator=(const mpi_window&) = delete;

    // Movable
    mpi_window(mpi_window&& other) noexcept
        : base_(other.base_)
        , size_(other.size_)
        , owns_memory_(other.owns_memory_)
#if DTL_ENABLE_MPI
        , win_(other.win_)
#endif
    {
#if DTL_ENABLE_MPI
        other.win_ = MPI_WIN_NULL;
#endif
        other.base_ = nullptr;
        other.size_ = 0;
        other.owns_memory_ = false;
    }

    mpi_window& operator=(mpi_window&& other) noexcept {
        if (this != &other) {
#if DTL_ENABLE_MPI
            if (win_ != MPI_WIN_NULL) {
                MPI_Win_free(&win_);
            }
            win_ = other.win_;
            other.win_ = MPI_WIN_NULL;
#endif
            base_ = other.base_;
            size_ = other.size_;
            owns_memory_ = other.owns_memory_;
            other.base_ = nullptr;
            other.size_ = 0;
            other.owns_memory_ = false;
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Query Methods
    // ------------------------------------------------------------------------

    /// @brief Get base address of local window memory
    [[nodiscard]] void* base() const noexcept { return base_; }

    /// @brief Get size of window in bytes
    [[nodiscard]] size_type size() const noexcept { return size_; }

    /// @brief Check if window is valid
    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_MPI
        return win_ != MPI_WIN_NULL;
#else
        return false;
#endif
    }

#if DTL_ENABLE_MPI
    /// @brief Get native MPI window handle
    [[nodiscard]] MPI_Win native_handle() const noexcept { return win_; }
#endif

    // ------------------------------------------------------------------------
    // RMA Operations
    // ------------------------------------------------------------------------

    /// @brief Put data to remote window
    /// @param origin Origin buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window (bytes)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> put_impl(
        const void* origin, size_type size, rank_t target, size_type target_offset) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Put(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            win_
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error, "MPI_Put failed");
        }
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Get data from remote window
    /// @param origin Origin buffer to receive data
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window (bytes)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> get_impl(
        void* origin, size_type size, rank_t target, size_type target_offset) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Get(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            win_
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error, "MPI_Get failed");
        }
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Synchronization Operations
    // ------------------------------------------------------------------------

    /// @brief Fence synchronization (active-target)
    /// @param assert_flags MPI assertion flags (e.g., MPI_MODE_NOPRECEDE)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> fence_impl(int assert_flags = 0) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_fence(assert_flags, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_fence failed");
        }
        return {};
#else
        (void)assert_flags;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Lock target for passive-target access
    /// @param target Target rank
    /// @param lock_type MPI lock type (MPI_LOCK_EXCLUSIVE or MPI_LOCK_SHARED)
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> lock_impl(rank_t target, int lock_type) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_lock(lock_type, target, 0, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_lock failed");
        }
        return {};
#else
        (void)target; (void)lock_type;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Unlock target
    /// @param target Target rank
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> unlock_impl(rank_t target) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_unlock(target, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_unlock failed");
        }
        return {};
#else
        (void)target;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Lock all targets for passive-target access
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> lock_all_impl() {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_lock_all(0, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_lock_all failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Unlock all targets
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> unlock_all_impl() {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_unlock_all(win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_unlock_all failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Flush operations to target
    /// @param target Target rank
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_impl(rank_t target) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_flush(target, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_flush failed");
        }
        return {};
#else
        (void)target;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Flush operations to all targets
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_all_impl() {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_flush_all(win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_flush_all failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Flush local operations to target
    /// @param target Target rank
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_local_impl(rank_t target) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_flush_local(target, win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_flush_local failed");
        }
        return {};
#else
        (void)target;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Flush local operations to all targets
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> flush_local_all_impl() {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        int mpi_result = MPI_Win_flush_local_all(win_);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Win_flush_local_all failed");
        }
        return {};
#else
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Atomic Operations
    // ------------------------------------------------------------------------

    /// @brief Accumulate operation
    /// @param origin Origin buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @param op MPI operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> accumulate_impl(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_reduce_op op) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        MPI_Op mpi_op = reduce_op_to_mpi(op);
        if (mpi_op == MPI_OP_NULL) {
            return make_error<void>(status_code::invalid_argument, "Invalid reduce op");
        }

        int mpi_result = MPI_Accumulate(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            mpi_op,
            win_
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Accumulate failed");
        }
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)op;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Fetch and operation
    /// @param origin Origin buffer
    /// @param result_buf Result buffer for old value
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @param op MPI operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> fetch_and_op_impl(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        MPI_Op mpi_op = reduce_op_to_mpi(op);
        if (mpi_op == MPI_OP_NULL) {
            return make_error<void>(status_code::invalid_argument, "Invalid reduce op");
        }

        // MPI_Fetch_and_op only works with single elements, use Get_accumulate for larger
        if (size <= sizeof(long long)) {
            int mpi_result = MPI_Fetch_and_op(
                origin,
                result_buf,
                MPI_BYTE,
                target,
                static_cast<MPI_Aint>(target_offset),
                mpi_op,
                win_
            );

            if (mpi_result != MPI_SUCCESS) {
                return make_error<void>(status_code::communication_error,
                                        "MPI_Fetch_and_op failed");
            }
        } else {
            // Fall back to Get_accumulate for larger data
            int mpi_result = MPI_Get_accumulate(
                origin,
                static_cast<int>(size),
                MPI_BYTE,
                result_buf,
                static_cast<int>(size),
                MPI_BYTE,
                target,
                static_cast<MPI_Aint>(target_offset),
                static_cast<int>(size),
                MPI_BYTE,
                mpi_op,
                win_
            );

            if (mpi_result != MPI_SUCCESS) {
                return make_error<void>(status_code::communication_error,
                                        "MPI_Get_accumulate failed");
            }
        }
        return {};
#else
        (void)origin; (void)result_buf; (void)size;
        (void)target; (void)target_offset; (void)op;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Compare and swap
    /// @param origin Origin buffer (new value)
    /// @param compare Compare buffer (expected value)
    /// @param result_buf Result buffer for old value
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> compare_and_swap_impl(
        const void* origin, const void* compare, void* result_buf,
        size_type size, rank_t target, size_type target_offset) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        // MPI_Compare_and_swap only works with single predefined datatypes
        // For simplicity, we require size to match a basic type
        MPI_Datatype dt;
        if (size == sizeof(int)) {
            dt = MPI_INT;
        } else if (size == sizeof(long)) {
            dt = MPI_LONG;
        } else if (size == sizeof(long long)) {
            dt = MPI_LONG_LONG;
        } else {
            return make_error<void>(status_code::invalid_argument,
                                    "Unsupported size for compare_and_swap");
        }

        int mpi_result = MPI_Compare_and_swap(
            origin,
            compare,
            result_buf,
            dt,
            target,
            static_cast<MPI_Aint>(target_offset),
            win_
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Compare_and_swap failed");
        }
        return {};
#else
        (void)origin; (void)compare; (void)result_buf;
        (void)size; (void)target; (void)target_offset;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Get-accumulate operation
    /// @param origin Origin buffer
    /// @param result_buf Result buffer for old values
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window
    /// @param op MPI operation
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> get_accumulate_impl(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }

        MPI_Op mpi_op = reduce_op_to_mpi(op);
        if (mpi_op == MPI_OP_NULL) {
            return make_error<void>(status_code::invalid_argument, "Invalid reduce op");
        }

        int mpi_result = MPI_Get_accumulate(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            result_buf,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            mpi_op,
            win_
        );

        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error,
                                    "MPI_Get_accumulate failed");
        }
        return {};
#else
        (void)origin; (void)result_buf; (void)size;
        (void)target; (void)target_offset; (void)op;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Non-Blocking RMA Operations (Request-Based)
    // ------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    /// @brief Non-blocking put data to remote window
    /// @param origin Origin buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window (bytes)
    /// @param request Output MPI_Request handle
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> rput_impl(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, MPI_Request* request) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }
        if (request == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null request");
        }

        int mpi_result = MPI_Rput(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            win_,
            request
        );

        if (mpi_result != MPI_SUCCESS) {
            *request = MPI_REQUEST_NULL;
            return make_error<void>(status_code::communication_error, "MPI_Rput failed");
        }
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)request;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Non-blocking get data from remote window
    /// @param origin Origin buffer to receive data
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param target_offset Offset in target window (bytes)
    /// @param request Output MPI_Request handle
    /// @return Result indicating success or failure
    [[nodiscard]] result<void> rget_impl(
        void* origin, size_type size, rank_t target,
        size_type target_offset, MPI_Request* request) {
#if DTL_ENABLE_MPI
        if (win_ == MPI_WIN_NULL) {
            return make_error<void>(status_code::invalid_state, "Invalid window");
        }
        if (request == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null request");
        }

        int mpi_result = MPI_Rget(
            origin,
            static_cast<int>(size),
            MPI_BYTE,
            target,
            static_cast<MPI_Aint>(target_offset),
            static_cast<int>(size),
            MPI_BYTE,
            win_,
            request
        );

        if (mpi_result != MPI_SUCCESS) {
            *request = MPI_REQUEST_NULL;
            return make_error<void>(status_code::communication_error, "MPI_Rget failed");
        }
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset; (void)request;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Test if a request has completed
    /// @param request The MPI request to test
    /// @param completed Output: true if completed
    /// @return Result indicating success or failure
    [[nodiscard]] static result<bool> test_request(MPI_Request* request) {
#if DTL_ENABLE_MPI
        if (request == nullptr) {
            return make_error<bool>(status_code::invalid_argument, "Null request");
        }
        if (*request == MPI_REQUEST_NULL) {
            return true;  // Already completed
        }

        int flag = 0;
        int mpi_result = MPI_Test(request, &flag, MPI_STATUS_IGNORE);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<bool>(status_code::communication_error, "MPI_Test failed");
        }
        return flag != 0;
#else
        (void)request;
        return make_error<bool>(status_code::not_supported, "MPI not enabled");
#endif
    }

    /// @brief Wait for a request to complete
    /// @param request The MPI request to wait on
    /// @return Result indicating success or failure
    [[nodiscard]] static result<void> wait_request(MPI_Request* request) {
#if DTL_ENABLE_MPI
        if (request == nullptr) {
            return make_error<void>(status_code::invalid_argument, "Null request");
        }
        if (*request == MPI_REQUEST_NULL) {
            return {};  // Already completed
        }

        int mpi_result = MPI_Wait(request, MPI_STATUS_IGNORE);
        if (mpi_result != MPI_SUCCESS) {
            return make_error<void>(status_code::communication_error, "MPI_Wait failed");
        }
        return {};
#else
        (void)request;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }
#endif  // DTL_ENABLE_MPI

private:
#if DTL_ENABLE_MPI
    /// @brief Construct from MPI window
    mpi_window(MPI_Win win, void* base, size_type size, bool owns_memory)
        : base_(base)
        , size_(size)
        , owns_memory_(owns_memory)
        , win_(win) {}

    /// @brief Convert rma_reduce_op to MPI_Op
    static MPI_Op reduce_op_to_mpi(rma_reduce_op op) {
        switch (op) {
            case rma_reduce_op::sum:     return MPI_SUM;
            case rma_reduce_op::prod:    return MPI_PROD;
            case rma_reduce_op::min:     return MPI_MIN;
            case rma_reduce_op::max:     return MPI_MAX;
            case rma_reduce_op::band:    return MPI_BAND;
            case rma_reduce_op::bor:     return MPI_BOR;
            case rma_reduce_op::bxor:    return MPI_BXOR;
            case rma_reduce_op::replace: return MPI_REPLACE;
            case rma_reduce_op::no_op:   return MPI_NO_OP;
            default:                     return MPI_OP_NULL;
        }
    }
#endif

    void* base_ = nullptr;
    size_type size_ = 0;
    bool owns_memory_ = false;
#if DTL_ENABLE_MPI
    MPI_Win win_ = MPI_WIN_NULL;
#endif
};

// ============================================================================
// MPI Window Implementation Adapter
// ============================================================================

/// @brief MPI window implementation of memory_window_impl
/// @details Adapts dtl::mpi::mpi_window to the memory_window_impl interface.
///          This is Layer 2 of the two-layer backend pattern, providing
///          a uniform interface for use with memory_window RAII wrapper.
///
/// @par Lifecycle
/// This adapter holds a non-owning reference to an mpi_window. The underlying
/// mpi_window object must outlive this adapter. The adapter does NOT own or
/// manage the MPI_Win lifecycle.
class mpi_window_impl : public dtl::memory_window_impl {
public:
    /// @brief Construct from a mpi_window reference
    /// @param win Reference to the underlying MPI window (must outlive this object)
    explicit mpi_window_impl(mpi_window& win) : win_(win) {}

    // -------------------------------------------------------------------------
    // Query Methods
    // -------------------------------------------------------------------------

    [[nodiscard]] void* base() const noexcept override {
        return win_.base();
    }

    [[nodiscard]] size_type size() const noexcept override {
        return win_.size();
    }

    [[nodiscard]] bool valid() const noexcept override {
        return win_.valid();
    }

    [[nodiscard]] void* native_handle() const noexcept override {
#if DTL_ENABLE_MPI
        // MPI_Win is a pointer type (ompi_win_t*), so just cast to void*
        return static_cast<void*>(win_.native_handle());
#else
        return nullptr;
#endif
    }

    // -------------------------------------------------------------------------
    // Synchronization Operations
    // -------------------------------------------------------------------------

    [[nodiscard]] result<void> fence(int assert_flags) override {
        return win_.fence_impl(assert_flags);
    }

    [[nodiscard]] result<void> lock_all() override {
        return win_.lock_all_impl();
    }

    [[nodiscard]] result<void> unlock_all() override {
        return win_.unlock_all_impl();
    }

    [[nodiscard]] result<void> lock(rank_t target, rma_lock_mode mode) override {
#if DTL_ENABLE_MPI
        // Convert rma_lock_mode to MPI lock type
        int lock_type = (mode == rma_lock_mode::exclusive) ? MPI_LOCK_EXCLUSIVE : MPI_LOCK_SHARED;
        return win_.lock_impl(target, lock_type);
#else
        (void)target; (void)mode;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    [[nodiscard]] result<void> unlock(rank_t target) override {
        return win_.unlock_impl(target);
    }

    [[nodiscard]] result<void> flush(rank_t target) override {
        return win_.flush_impl(target);
    }

    [[nodiscard]] result<void> flush_all() override {
        return win_.flush_all_impl();
    }

    [[nodiscard]] result<void> flush_local(rank_t target) override {
        return win_.flush_local_impl(target);
    }

    [[nodiscard]] result<void> flush_local_all() override {
        return win_.flush_local_all_impl();
    }

    // -------------------------------------------------------------------------
    // Data Transfer Operations
    // -------------------------------------------------------------------------

    [[nodiscard]] result<void> put(
        const void* origin, size_type size, rank_t target, size_type target_offset) override {
        return win_.put_impl(origin, size, target, target_offset);
    }

    [[nodiscard]] result<void> get(
        void* origin, size_type size, rank_t target, size_type target_offset) override {
        return win_.get_impl(origin, size, target, target_offset);
    }

    [[nodiscard]] result<void> accumulate(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_reduce_op op) override {
        return win_.accumulate_impl(origin, size, target, target_offset, op);
    }

    [[nodiscard]] result<void> fetch_and_op(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
        return win_.fetch_and_op_impl(origin, result_buf, size, target, target_offset, op);
    }

    [[nodiscard]] result<void> compare_and_swap(
        const void* origin, const void* compare, void* result_buf,
        size_type size, rank_t target, size_type target_offset) override {
        return win_.compare_and_swap_impl(origin, compare, result_buf, size, target, target_offset);
    }

    [[nodiscard]] result<void> get_accumulate(
        const void* origin, void* result_buf, size_type size,
        rank_t target, size_type target_offset, rma_reduce_op op) override {
        return win_.get_accumulate_impl(origin, result_buf, size, target, target_offset, op);
    }

    // -------------------------------------------------------------------------
    // Non-Blocking RMA Operations
    // -------------------------------------------------------------------------

    [[nodiscard]] result<void> async_put(
        const void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) override {
#if DTL_ENABLE_MPI
        // Allocate request storage
        auto* mpi_req = new MPI_Request(MPI_REQUEST_NULL);
        auto res = win_.rput_impl(origin, size, target, target_offset, mpi_req);
        if (res.has_error()) {
            delete mpi_req;
            request.handle = nullptr;
            request.completed = true;
            return res;
        }
        request.handle = mpi_req;
        request.completed = false;
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        request.completed = true;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    [[nodiscard]] result<void> async_get(
        void* origin, size_type size, rank_t target,
        size_type target_offset, rma_request_handle& request) override {
#if DTL_ENABLE_MPI
        // Allocate request storage
        auto* mpi_req = new MPI_Request(MPI_REQUEST_NULL);
        auto res = win_.rget_impl(origin, size, target, target_offset, mpi_req);
        if (res.has_error()) {
            delete mpi_req;
            request.handle = nullptr;
            request.completed = true;
            return res;
        }
        request.handle = mpi_req;
        request.completed = false;
        return {};
#else
        (void)origin; (void)size; (void)target; (void)target_offset;
        request.completed = true;
        return make_error<void>(status_code::not_supported, "MPI not enabled");
#endif
    }

    [[nodiscard]] result<bool> test_async(rma_request_handle& request) override {
#if DTL_ENABLE_MPI
        if (request.completed || request.handle == nullptr) {
            return true;
        }
        auto* mpi_req = static_cast<MPI_Request*>(request.handle);
        auto res = mpi_window::test_request(mpi_req);
        if (res.has_error()) {
            return res.error();
        }
        if (res.value()) {
            request.completed = true;
            delete mpi_req;
            request.handle = nullptr;
        }
        return res.value();
#else
        return request.completed;
#endif
    }

    [[nodiscard]] result<void> wait_async(rma_request_handle& request) override {
#if DTL_ENABLE_MPI
        if (request.completed || request.handle == nullptr) {
            return {};
        }
        auto* mpi_req = static_cast<MPI_Request*>(request.handle);
        auto res = mpi_window::wait_request(mpi_req);
        delete mpi_req;
        request.handle = nullptr;
        request.completed = true;
        return res;
#else
        request.completed = true;
        return {};
#endif
    }

private:
    /// @brief Non-owning reference to the underlying MPI window
    mpi_window& win_;
};

}  // namespace mpi
}  // namespace dtl
