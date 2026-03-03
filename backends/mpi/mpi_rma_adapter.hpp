// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_rma_adapter.hpp
/// @brief Concept-compliant MPI RMA adapter (Layer 2 - throws on error)
/// @details Wraps mpi_window and mpi_comm_adapter to satisfy FullRmaCommunicator concept.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/mpi/mpi_window.hpp>

#include <stdexcept>
#include <unordered_map>
#include <memory>

namespace dtl {
namespace mpi {

// ============================================================================
// MPI RMA Adapter
// ============================================================================

/// @brief Concept-compliant MPI RMA adapter
/// @details Provides void-returning methods that throw on error.
///          Satisfies FullRmaCommunicator concept (RmaCommunicator +
///          PassiveTargetRmaCommunicator + AtomicRmaCommunicator).
class mpi_rma_adapter {
public:
    using size_type = dtl::size_type;

    /// @brief Construct with world communicator
    mpi_rma_adapter()
        : comm_adapter_() {}

    /// @brief Construct with specific communicator
    /// @param comm Reference to mpi_communicator (must outlive adapter)
    explicit mpi_rma_adapter(mpi_communicator& comm)
        : comm_adapter_(comm) {}

    /// @brief Destructor — frees all owned windows
    ~mpi_rma_adapter() {
        for (auto& [handle, win_ptr] : windows_) {
            delete win_ptr;
        }
        windows_.clear();
    }

    // Non-copyable (owns raw window pointers)
    mpi_rma_adapter(const mpi_rma_adapter&) = delete;
    mpi_rma_adapter& operator=(const mpi_rma_adapter&) = delete;

    // Movable
    mpi_rma_adapter(mpi_rma_adapter&& other) noexcept
        : comm_adapter_(std::move(other.comm_adapter_)),
          windows_(std::move(other.windows_)) {
        other.windows_.clear();
    }
    mpi_rma_adapter& operator=(mpi_rma_adapter&& other) noexcept {
        if (this != &other) {
            for (auto& [handle, win_ptr] : windows_) {
                delete win_ptr;
            }
            comm_adapter_ = std::move(other.comm_adapter_);
            windows_ = std::move(other.windows_);
            other.windows_.clear();
        }
        return *this;
    }

    // ------------------------------------------------------------------------
    // Communicator Interface (delegated to mpi_comm_adapter)
    // ------------------------------------------------------------------------

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept {
        return comm_adapter_.rank();
    }

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept {
        return comm_adapter_.size();
    }

    /// @brief Blocking send
    void send(const void* buf, size_type count, rank_t dest, int tag) {
        comm_adapter_.send(buf, count, dest, tag);
    }

    /// @brief Blocking receive
    void recv(void* buf, size_type count, rank_t source, int tag) {
        comm_adapter_.recv(buf, count, source, tag);
    }

    /// @brief Non-blocking send
    [[nodiscard]] request_handle isend(const void* buf, size_type count,
                                        rank_t dest, int tag) {
        return comm_adapter_.isend(buf, count, dest, tag);
    }

    /// @brief Non-blocking receive
    [[nodiscard]] request_handle irecv(void* buf, size_type count,
                                        rank_t source, int tag) {
        return comm_adapter_.irecv(buf, count, source, tag);
    }

    /// @brief Wait for non-blocking operation
    void wait(request_handle& req) {
        comm_adapter_.wait(req);
    }

    /// @brief Test if non-blocking operation completed
    [[nodiscard]] bool test(request_handle& req) {
        return comm_adapter_.test(req);
    }

    // ------------------------------------------------------------------------
    // Window Management
    // ------------------------------------------------------------------------

    /// @brief Create a window from existing memory
    /// @param base Pointer to local memory
    /// @param size Size in bytes
    /// @return Window handle
    /// @throws communication_error on failure
    [[nodiscard]] window_handle create_window(void* base, size_type size) {
        auto result = mpi_window::create(base, size, comm_adapter_.underlying());
        if (!result) {
            throw communication_error("Failed to create MPI window");
        }

        auto* win_ptr = new mpi_window(std::move(*result));
        window_handle handle{static_cast<void*>(win_ptr)};
        windows_[handle.handle] = win_ptr;
        return handle;
    }

    /// @brief Free a window
    /// @param win Window handle
    /// @throws communication_error on failure
    void free_window(window_handle& win) {
        if (!win.valid()) {
            return;
        }

        auto it = windows_.find(win.handle);
        if (it != windows_.end()) {
            delete it->second;
            windows_.erase(it);
        }
        win.handle = nullptr;
    }

    // ------------------------------------------------------------------------
    // One-Sided Data Transfer
    // ------------------------------------------------------------------------

    /// @brief Put data to remote window
    /// @param buf Source buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param offset Offset in target window
    /// @param win Window handle
    /// @throws communication_error on failure
    void put(const void* buf, size_type size, rank_t target,
             size_type offset, window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->put_impl(buf, size, target, offset);
        if (!result) {
            throw communication_error("MPI put failed");
        }
    }

    /// @brief Get data from remote window
    /// @param buf Destination buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param offset Offset in target window
    /// @param win Window handle
    /// @throws communication_error on failure
    void get(void* buf, size_type size, rank_t target,
             size_type offset, window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->get_impl(buf, size, target, offset);
        if (!result) {
            throw communication_error("MPI get failed");
        }
    }

    // ------------------------------------------------------------------------
    // Active-Target Synchronization
    // ------------------------------------------------------------------------

    /// @brief Fence synchronization
    /// @param win Window handle
    /// @throws communication_error on failure
    void fence(window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->fence_impl(0);
        if (!result) {
            throw communication_error("MPI fence failed");
        }
    }

    // ------------------------------------------------------------------------
    // Remote Completion
    // ------------------------------------------------------------------------

    /// @brief Flush operations to target
    /// @param target Target rank
    /// @param win Window handle
    /// @throws communication_error on failure
    void flush(rank_t target, window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->flush_impl(target);
        if (!result) {
            throw communication_error("MPI flush failed");
        }
    }

    /// @brief Flush operations to all targets
    /// @param win Window handle
    /// @throws communication_error on failure
    void flush_all(window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->flush_all_impl();
        if (!result) {
            throw communication_error("MPI flush_all failed");
        }
    }

    // ------------------------------------------------------------------------
    // Passive-Target Synchronization
    // ------------------------------------------------------------------------

    /// @brief Lock target window
    /// @param target Target rank
    /// @param mode Lock mode
    /// @param win Window handle
    /// @throws communication_error on failure
    void lock(rank_t target, rma_lock_mode mode, window_handle& win) {
#if DTL_ENABLE_MPI
        auto* mpi_win = get_window(win);
        int lock_type = (mode == rma_lock_mode::exclusive)
                        ? MPI_LOCK_EXCLUSIVE : MPI_LOCK_SHARED;
        auto result = mpi_win->lock_impl(target, lock_type);
        if (!result) {
            throw communication_error("MPI lock failed");
        }
#else
        (void)target; (void)mode; (void)win;
        throw communication_error("MPI not enabled");
#endif
    }

    /// @brief Unlock target window
    /// @param target Target rank
    /// @param win Window handle
    /// @throws communication_error on failure
    void unlock(rank_t target, window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->unlock_impl(target);
        if (!result) {
            throw communication_error("MPI unlock failed");
        }
    }

    /// @brief Lock all windows
    /// @param win Window handle
    /// @throws communication_error on failure
    void lock_all(window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->lock_all_impl();
        if (!result) {
            throw communication_error("MPI lock_all failed");
        }
    }

    /// @brief Unlock all windows
    /// @param win Window handle
    /// @throws communication_error on failure
    void unlock_all(window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->unlock_all_impl();
        if (!result) {
            throw communication_error("MPI unlock_all failed");
        }
    }

    // ------------------------------------------------------------------------
    // Atomic Operations
    // ------------------------------------------------------------------------

    /// @brief Accumulate operation
    /// @param origin Origin buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param offset Offset in target window
    /// @param op Reduction operation
    /// @param win Window handle
    /// @throws communication_error on failure
    void accumulate(const void* origin, size_type size, rank_t target,
                    size_type offset, rma_reduce_op op, window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->accumulate_impl(origin, size, target, offset, op);
        if (!result) {
            throw communication_error("MPI accumulate failed");
        }
    }

    /// @brief Fetch and operation
    /// @param origin Origin buffer
    /// @param result_buf Result buffer
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param offset Offset in target window
    /// @param op Reduction operation
    /// @param win Window handle
    /// @throws communication_error on failure
    void fetch_and_op(const void* origin, void* result_buf, size_type size,
                      rank_t target, size_type offset, rma_reduce_op op,
                      window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->fetch_and_op_impl(origin, result_buf, size,
                                                  target, offset, op);
        if (!result) {
            throw communication_error("MPI fetch_and_op failed");
        }
    }

    /// @brief Compare and swap
    /// @param origin Origin buffer (new value)
    /// @param compare Compare buffer (expected value)
    /// @param result_buf Result buffer (receives old value)
    /// @param size Data size in bytes
    /// @param target Target rank
    /// @param offset Offset in target window
    /// @param win Window handle
    /// @throws communication_error on failure
    void compare_and_swap(const void* origin, const void* compare,
                          void* result_buf, size_type size,
                          rank_t target, size_type offset,
                          window_handle& win) {
        auto* mpi_win = get_window(win);
        auto result = mpi_win->compare_and_swap_impl(origin, compare, result_buf,
                                                      size, target, offset);
        if (!result) {
            throw communication_error("MPI compare_and_swap failed");
        }
    }

    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------

    /// @brief Get underlying communicator adapter
    [[nodiscard]] mpi_comm_adapter& comm() noexcept {
        return comm_adapter_;
    }

    /// @brief Get underlying communicator adapter (const)
    [[nodiscard]] const mpi_comm_adapter& comm() const noexcept {
        return comm_adapter_;
    }

private:
    /// @brief Get mpi_window from handle
    /// @throws communication_error if invalid
    mpi_window* get_window(window_handle& win) {
        if (!win.valid()) {
            throw communication_error("Invalid window handle");
        }
        auto it = windows_.find(win.handle);
        if (it == windows_.end()) {
            throw communication_error("Window not found");
        }
        return it->second;
    }

    mpi_comm_adapter comm_adapter_;
    std::unordered_map<void*, mpi_window*> windows_;
};

// ============================================================================
// Concept Verification
// ============================================================================

// Note: The static_assert checks will only pass when DTL_ENABLE_MPI is defined
// because the concepts require actual MPI functionality.
// When MPI is disabled, the adapter still compiles but won't satisfy the concepts
// since operations will throw "not supported" errors.

#if DTL_ENABLE_MPI
static_assert(RmaCommunicator<mpi_rma_adapter>,
              "mpi_rma_adapter must satisfy RmaCommunicator concept");
static_assert(PassiveTargetRmaCommunicator<mpi_rma_adapter>,
              "mpi_rma_adapter must satisfy PassiveTargetRmaCommunicator concept");
static_assert(AtomicRmaCommunicator<mpi_rma_adapter>,
              "mpi_rma_adapter must satisfy AtomicRmaCommunicator concept");
static_assert(FullRmaCommunicator<mpi_rma_adapter>,
              "mpi_rma_adapter must satisfy FullRmaCommunicator concept");
#endif

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get a world RMA adapter
/// @return RMA adapter wrapping MPI_COMM_WORLD
[[nodiscard]] inline mpi_rma_adapter world_rma_adapter() {
    return mpi_rma_adapter{};
}

}  // namespace mpi
}  // namespace dtl
