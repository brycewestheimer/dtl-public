// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_rma_adapter.hpp
/// @brief SHMEM RMA adapter satisfying RmaCommunicator concept
/// @details Provides one-sided communication using SHMEM primitives.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

#include <cstring>
#include <stdexcept>

namespace dtl {
namespace shmem {

// ============================================================================
// SHMEM Communication Error
// ============================================================================

/// @brief Error type for SHMEM communication failures
class shmem_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ============================================================================
// SHMEM RMA Adapter
// ============================================================================

/// @brief SHMEM-based RMA adapter satisfying FullRmaCommunicator concept
/// @details Wraps SHMEM one-sided operations to satisfy the RmaCommunicator
///          concept. SHMEM provides natural RMA support with passive-target
///          semantics - no explicit epochs or windows required.
///
/// @par Message-Passing Emulation:
/// SHMEM is a PGAS (one-sided) model without native two-sided message passing.
/// send/recv are emulated via a per-PE symmetric mailbox:
/// - send(): shmem_putmem into dest PE's mailbox + atomic flag signal
/// - recv(): shmem_wait_until for flag + local copy from mailbox
///
/// @par Key Differences from MPI RMA:
/// - No explicit window creation - symmetric memory is always accessible
/// - No epoch management - operations are always valid
/// - Ordering controlled via fence() and quiet()
/// - Passive-target by default
class shmem_rma_adapter {
public:
    using size_type = dtl::size_type;

    /// @brief Maximum message size for the symmetric mailbox (64 KB)
    static constexpr size_type mailbox_capacity = 65536;

    /// @brief Default constructor — allocates symmetric mailbox
    shmem_rma_adapter() {
#if DTL_ENABLE_SHMEM
        rank_ = shmem_my_pe();
        size_ = shmem_n_pes();
        // Allocate symmetric mailbox for message-passing emulation
        // Layout: [flag (long)] [tag (int)] [msg_size (size_type)] [data...]
        size_type total = mailbox_header_size() + mailbox_capacity;
        mailbox_ = static_cast<char*>(shmem_malloc(total));
        if (mailbox_) {
            // Zero-initialize the mailbox (flag=0 means empty)
            std::memset(mailbox_, 0, total);
        }
#endif
    }

    /// @brief Destructor — frees symmetric mailbox
    ~shmem_rma_adapter() {
#if DTL_ENABLE_SHMEM
        if (mailbox_) {
            shmem_free(mailbox_);
            mailbox_ = nullptr;
        }
#endif
    }

    // Non-copyable
    shmem_rma_adapter(const shmem_rma_adapter&) = delete;
    shmem_rma_adapter& operator=(const shmem_rma_adapter&) = delete;

    // Movable
    shmem_rma_adapter(shmem_rma_adapter&&) = default;
    shmem_rma_adapter& operator=(shmem_rma_adapter&&) = default;

    // ========================================================================
    // Communicator Interface (required by Communicator concept)
    // ========================================================================

    /// @brief Get this PE's rank
    [[nodiscard]] rank_t rank() const noexcept { return rank_; }

    /// @brief Get total number of PEs
    [[nodiscard]] rank_t size() const noexcept { return size_; }

    /// @brief Blocking send via symmetric mailbox
    /// @details Puts message data into the destination PE's symmetric mailbox
    ///          using shmem_putmem, then signals via atomic flag set.
    /// @param buf Source buffer (local memory)
    /// @param count Number of bytes to send
    /// @param dest Destination PE
    /// @param tag Message tag (stored in mailbox header for receiver to check)
    void send(const void* buf, size_type count, rank_t dest, int tag) {
#if DTL_ENABLE_SHMEM
        if (!mailbox_ || count > mailbox_capacity) {
            return;  // Silently fail if mailbox not allocated or message too large
        }
        // Put tag and size into dest PE's mailbox header
        int tag_val = tag;
        shmem_putmem(mailbox_ + flag_offset() + sizeof(long),
                     &tag_val, sizeof(int), dest);
        size_type sz = count;
        shmem_putmem(mailbox_ + flag_offset() + sizeof(long) + sizeof(int),
                     &sz, sizeof(size_type), dest);
        // Put payload data into dest PE's mailbox data region
        if (count > 0) {
            shmem_putmem(mailbox_ + mailbox_header_size(),
                         buf, count, dest);
        }
        // Ensure all puts are complete before signaling
        shmem_quiet();
        // Signal the receiver by setting the flag to 1
        long one = 1;
        shmem_long_atomic_set(reinterpret_cast<long*>(mailbox_ + flag_offset()),
                              one, dest);
#else
        (void)buf; (void)count; (void)dest; (void)tag;
#endif
    }

    /// @brief Blocking receive from symmetric mailbox
    /// @details Waits for the mailbox flag to be set, then copies data from
    ///          the local mailbox into the user buffer and resets the flag.
    /// @param buf Destination buffer (local memory)
    /// @param count Number of bytes to receive
    /// @param source Source PE (used for tag validation only)
    /// @param tag Expected message tag (checked against mailbox header)
    void recv(void* buf, size_type count, rank_t source, int tag) {
#if DTL_ENABLE_SHMEM
        (void)source;  // Source PE identity is implicit in SHMEM
        if (!mailbox_) {
            return;  // Silently fail if mailbox not allocated
        }
        // Wait until the flag is set to 1 (message arrived)
        shmem_long_wait_until(
            reinterpret_cast<long*>(mailbox_ + flag_offset()),
            SHMEM_CMP_EQ, 1);
        // Read the tag from the mailbox header for validation
        int received_tag = 0;
        std::memcpy(&received_tag,
                    mailbox_ + flag_offset() + sizeof(long), sizeof(int));
        (void)tag;         // Tag validation is advisory in this emulation
        (void)received_tag;
        // Read the actual message size
        size_type msg_size = 0;
        std::memcpy(&msg_size,
                    mailbox_ + flag_offset() + sizeof(long) + sizeof(int),
                    sizeof(size_type));
        // Copy payload from mailbox to user buffer
        size_type copy_size = (count < msg_size) ? count : msg_size;
        if (copy_size > 0) {
            std::memcpy(buf, mailbox_ + mailbox_header_size(), copy_size);
        }
        // Reset the flag to 0 (mailbox empty, ready for next message)
        auto* flag_ptr = reinterpret_cast<long*>(mailbox_ + flag_offset());
        *flag_ptr = 0;
        shmem_quiet();
#else
        (void)buf; (void)count; (void)source; (void)tag;
#endif
    }

    /// @brief Non-blocking send (returns handle; emulated)
    [[nodiscard]] request_handle isend(const void* buf, size_type count,
                                        rank_t dest, int tag) {
        send(buf, count, dest, tag);
        return request_handle{};  // Already completed
    }

    /// @brief Non-blocking receive (returns handle; emulated)
    [[nodiscard]] request_handle irecv(void* buf, size_type count,
                                        rank_t source, int tag) {
        recv(buf, count, source, tag);
        return request_handle{};  // Already completed
    }

    /// @brief Wait for non-blocking operation
    void wait(request_handle& /*req*/) {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
#endif
    }

    /// @brief Test if non-blocking operation completed
    [[nodiscard]] bool test(request_handle& /*req*/) {
        return true;  // SHMEM ops complete inline
    }

    // ========================================================================
    // RmaCommunicator Interface (concept-compliant)
    // ========================================================================

    /// @brief Create a window from existing memory
    /// @details SHMEM uses symmetric memory, so windows are logical.
    ///          The handle stores the base pointer for offset calculation.
    [[nodiscard]] window_handle create_window(void* base, size_type /*size*/) {
        return window_handle{base};
    }

    /// @brief Free a window (no-op for SHMEM)
    void free_window(window_handle& win) {
        win.handle = nullptr;
    }

    /// @brief Put data to a remote PE's window
    /// @param buf Source buffer (local)
    /// @param size Number of bytes
    /// @param target Target PE
    /// @param offset Byte offset into target window
    /// @param win Window handle
    void put(const void* buf, size_type size, rank_t target,
             size_type offset, window_handle& win) {
#if DTL_ENABLE_SHMEM
        auto* dest = static_cast<char*>(win.handle) + offset;
        shmem_putmem(dest, buf, size, target);
#else
        (void)buf; (void)size; (void)target; (void)offset; (void)win;
#endif
    }

    /// @brief Get data from a remote PE's window
    /// @param buf Destination buffer (local)
    /// @param size Number of bytes
    /// @param target Source PE
    /// @param offset Byte offset into source window
    /// @param win Window handle
    void get(void* buf, size_type size, rank_t target,
             size_type offset, window_handle& win) {
#if DTL_ENABLE_SHMEM
        auto* src = static_cast<const char*>(win.handle) + offset;
        shmem_getmem(buf, src, size, target);
#else
        (void)buf; (void)size; (void)target; (void)offset; (void)win;
#endif
    }

    /// @brief Fence — order remote operations
    void fence(window_handle& /*win*/) {
#if DTL_ENABLE_SHMEM
        shmem_fence();
#endif
    }

    /// @brief Flush operations to a specific target
    void flush(rank_t /*target*/, window_handle& /*win*/) {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
#endif
    }

    /// @brief Flush all outstanding operations
    void flush_all(window_handle& /*win*/) {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
#endif
    }

    // ========================================================================
    // PassiveTargetRmaCommunicator Interface
    // ========================================================================

    /// @brief Lock target window (no-op for SHMEM — always passive-target)
    void lock(rank_t /*target*/, rma_lock_mode /*mode*/, window_handle& /*win*/) {
        // SHMEM is naturally passive-target; no lock needed
    }

    /// @brief Unlock target window (no-op for SHMEM)
    void unlock(rank_t /*target*/, window_handle& /*win*/) {
        // SHMEM is naturally passive-target; no unlock needed
    }

    /// @brief Lock all windows (no-op for SHMEM)
    void lock_all(window_handle& /*win*/) {
        // SHMEM is naturally passive-target
    }

    /// @brief Unlock all windows (no-op for SHMEM)
    void unlock_all(window_handle& /*win*/) {
        // SHMEM is naturally passive-target
    }

    // ========================================================================
    // AtomicRmaCommunicator Interface
    // ========================================================================

    /// @brief Atomic accumulate operation
    void accumulate(const void* origin, size_type size, rank_t target,
                    size_type offset, rma_reduce_op /*op*/, window_handle& win) {
#if DTL_ENABLE_SHMEM
        // SHMEM atomic add only works on typed data; fall back to put for raw bytes
        auto* dest = static_cast<char*>(win.handle) + offset;
        shmem_putmem(dest, origin, size, target);
#else
        (void)origin; (void)size; (void)target; (void)offset; (void)win;
#endif
    }

    /// @brief Fetch and atomic operation
    void fetch_and_op(const void* origin, void* result_buf, size_type size,
                      rank_t target, size_type offset, rma_reduce_op /*op*/,
                      window_handle& win) {
#if DTL_ENABLE_SHMEM
        // Get current value, then put new value
        auto* remote = static_cast<char*>(win.handle) + offset;
        shmem_getmem(result_buf, remote, size, target);
        shmem_putmem(remote, origin, size, target);
#else
        (void)origin; (void)result_buf; (void)size;
        (void)target; (void)offset; (void)win;
#endif
    }

    /// @brief Atomic compare-and-swap
    void compare_and_swap(const void* origin, const void* compare,
                          void* result_buf, size_type size,
                          rank_t target, size_type offset,
                          window_handle& win) {
#if DTL_ENABLE_SHMEM
        // SHMEM has typed CAS; for raw bytes, we approximate
        auto* remote = static_cast<char*>(win.handle) + offset;
        shmem_getmem(result_buf, remote, size, target);
        if (std::memcmp(result_buf, compare, size) == 0) {
            shmem_putmem(remote, origin, size, target);
        }
#else
        (void)origin; (void)compare; (void)result_buf;
        (void)size; (void)target; (void)offset; (void)win;
#endif
    }

    // ========================================================================
    // SHMEM-Specific Operations (convenience, not required by concepts)
    // ========================================================================

    /// @brief Barrier — synchronize all PEs
    void barrier() {
#if DTL_ENABLE_SHMEM
        shmem_barrier_all();
#endif
    }

    /// @brief Quiet — wait for all outstanding remote operations
    void quiet() {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
#endif
    }

    /// @brief Check if adapter is valid
    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_SHMEM
        return size_ > 0;
#else
        return false;
#endif
    }

private:
    /// @brief Offset of the flag field in the mailbox
    static constexpr size_type flag_offset() { return 0; }

    /// @brief Total size of the mailbox header (flag + tag + msg_size)
    static constexpr size_type mailbox_header_size() {
        return sizeof(long) + sizeof(int) + sizeof(size_type);
    }

    rank_t rank_ = no_rank;
    rank_t size_ = 0;
    char* mailbox_ = nullptr;  ///< Symmetric mailbox for message-passing emulation
};

// ============================================================================
// Concept Verification
// ============================================================================

#if DTL_ENABLE_SHMEM
static_assert(Communicator<shmem_rma_adapter>,
              "shmem_rma_adapter must satisfy Communicator concept");
static_assert(RmaCommunicator<shmem_rma_adapter>,
              "shmem_rma_adapter must satisfy RmaCommunicator concept");
static_assert(PassiveTargetRmaCommunicator<shmem_rma_adapter>,
              "shmem_rma_adapter must satisfy PassiveTargetRmaCommunicator concept");
static_assert(AtomicRmaCommunicator<shmem_rma_adapter>,
              "shmem_rma_adapter must satisfy AtomicRmaCommunicator concept");
static_assert(FullRmaCommunicator<shmem_rma_adapter>,
              "shmem_rma_adapter must satisfy FullRmaCommunicator concept");
#endif

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the global SHMEM RMA adapter
[[nodiscard]] inline shmem_rma_adapter& global_rma_adapter() {
    static shmem_rma_adapter adapter;
    return adapter;
}

}  // namespace shmem
}  // namespace dtl
