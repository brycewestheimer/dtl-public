// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shared_memory_communicator.hpp
/// @brief Shared memory communicator for intra-node communication
/// @details Provides efficient communication using shared memory regions
///          with POSIX shm_open/mmap on Linux/macOS.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <atomic>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dtl {
namespace shared_memory {

// ============================================================================
// Shared Memory Region
// ============================================================================

/// @brief A shared memory region accessible by multiple processes
/// @details On POSIX systems (Linux, macOS), uses shm_open + mmap for true
///          inter-process shared memory. Falls back to heap allocation on
///          unsupported platforms (Windows stub, or single-process testing).
class shared_region {
public:
    /// @brief Construct empty region
    shared_region() = default;

    /// @brief Construct region with given size (creator mode)
    /// @param size Size in bytes
    /// @param region_id Unique identifier for the region
    explicit shared_region(size_type size, size_type region_id = 0)
        : size_(size)
        , region_id_(region_id) {
#if defined(__linux__) || defined(__APPLE__)
        name_ = "/dtl_shm_" + std::to_string(region_id);
        fd_ = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ < 0) {
            // Fallback to heap if shm_open fails (e.g., no /dev/shm)
            heap_fallback_ = std::make_unique<char[]>(size);
            data_ = heap_fallback_.get();
            return;
        }
        if (ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            close(fd_);
            shm_unlink(name_.c_str());
            fd_ = -1;
            heap_fallback_ = std::make_unique<char[]>(size);
            data_ = heap_fallback_.get();
            return;
        }
        data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            close(fd_);
            shm_unlink(name_.c_str());
            fd_ = -1;
            heap_fallback_ = std::make_unique<char[]>(size);
            data_ = heap_fallback_.get();
            return;
        }
        is_creator_ = true;
#else
        // Windows stub / unsupported: heap fallback
        heap_fallback_ = std::make_unique<char[]>(size);
        data_ = heap_fallback_.get();
#endif
    }

    /// @brief Open an existing shared region (non-creator mode)
    /// @param region_id Unique identifier matching the creator
    /// @param size Expected size in bytes
    /// @return The opened shared region, or empty on failure
    [[nodiscard]] static shared_region open(size_type region_id, size_type size) {
        shared_region r;
        r.size_ = size;
        r.region_id_ = region_id;
#if defined(__linux__) || defined(__APPLE__)
        r.name_ = "/dtl_shm_" + std::to_string(region_id);
        r.fd_ = shm_open(r.name_.c_str(), O_RDWR, 0600);
        if (r.fd_ < 0) return r;
        r.data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd_, 0);
        if (r.data_ == MAP_FAILED) {
            r.data_ = nullptr;
            close(r.fd_);
            r.fd_ = -1;
        }
        r.is_creator_ = false;
#endif
        return r;
    }

    /// @brief Destructor — unmaps and optionally unlinks
    ~shared_region() {
#if defined(__linux__) || defined(__APPLE__)
        if (data_ && !heap_fallback_) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        if (is_creator_ && !name_.empty()) {
            shm_unlink(name_.c_str());
        }
#endif
    }

    // Non-copyable
    shared_region(const shared_region&) = delete;
    shared_region& operator=(const shared_region&) = delete;

    // Movable
    shared_region(shared_region&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , region_id_(other.region_id_)
        , heap_fallback_(std::move(other.heap_fallback_))
#if defined(__linux__) || defined(__APPLE__)
        , fd_(other.fd_)
        , name_(std::move(other.name_))
        , is_creator_(other.is_creator_)
#endif
    {
        other.data_ = nullptr;
        other.size_ = 0;
#if defined(__linux__) || defined(__APPLE__)
        other.fd_ = -1;
        other.is_creator_ = false;
#endif
    }

    shared_region& operator=(shared_region&& other) noexcept {
        if (this != &other) {
            // Clean up current
#if defined(__linux__) || defined(__APPLE__)
            if (data_ && !heap_fallback_) munmap(data_, size_);
            if (fd_ >= 0) close(fd_);
            if (is_creator_ && !name_.empty()) shm_unlink(name_.c_str());
#endif
            data_ = other.data_;
            size_ = other.size_;
            region_id_ = other.region_id_;
            heap_fallback_ = std::move(other.heap_fallback_);
#if defined(__linux__) || defined(__APPLE__)
            fd_ = other.fd_;
            name_ = std::move(other.name_);
            is_creator_ = other.is_creator_;
            other.fd_ = -1;
            other.is_creator_ = false;
#endif
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /// @brief Get pointer to region data
    [[nodiscard]] void* data() noexcept { return data_; }

    /// @brief Get const pointer to region data
    [[nodiscard]] const void* data() const noexcept { return data_; }

    /// @brief Get region size
    [[nodiscard]] size_type size() const noexcept { return size_; }

    /// @brief Get region ID
    [[nodiscard]] size_type region_id() const noexcept { return region_id_; }

    /// @brief Check if region is valid
    [[nodiscard]] bool valid() const noexcept { return data_ != nullptr; }

    /// @brief Check if backed by real shared memory (not heap fallback)
    [[nodiscard]] bool is_shared() const noexcept {
#if defined(__linux__) || defined(__APPLE__)
        return data_ != nullptr && !heap_fallback_;
#else
        return false;
#endif
    }

private:
    void* data_ = nullptr;
    size_type size_ = 0;
    size_type region_id_ = 0;
    std::unique_ptr<char[]> heap_fallback_;

#if defined(__linux__) || defined(__APPLE__)
    int fd_ = -1;
    std::string name_;
    bool is_creator_ = false;
#endif
};

// ============================================================================
// Synchronization Primitives
// ============================================================================

/// @brief Shared memory barrier for process synchronization
class shared_barrier {
public:
    /// @brief Construct barrier for given count
    /// @param count Number of participants
    explicit shared_barrier(size_type count)
        : count_(count)
        , arrived_(0)
        , generation_(0) {}

    /// @brief Wait at barrier
    /// @details Uses C++20 std::atomic::wait()/notify_all() instead of a bare
    ///          spin loop, allowing the OS to efficiently block the thread
    ///          until the generation counter changes.
    void arrive_and_wait() {
        size_type gen = generation_.load(std::memory_order_acquire);
        size_type arrived = arrived_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (arrived == count_) {
            // Last to arrive - reset and release
            arrived_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            generation_.notify_all();
        } else {
            // Wait for generation to change (C++20 atomic wait)
            generation_.wait(gen, std::memory_order_acquire);
        }
    }

private:
    size_type count_;
    std::atomic<size_type> arrived_;
    std::atomic<size_type> generation_;
};

// ============================================================================
// Mailbox for Point-to-Point Communication
// ============================================================================

/// @brief Per-rank mailbox header in shared memory
/// @details Layout: [flag (atomic<int>)] [tag (int)] [msg_size (size_type)] [data...]
struct alignas(64) mailbox_header {
    std::atomic<int> flag;  ///< 0 = empty, 1 = message ready
    int tag;                ///< Message tag
    size_type msg_size;     ///< Actual payload size in bytes
    // Payload data follows immediately after this header
};

// ============================================================================
// Shared Memory Communicator
// ============================================================================

/// @brief Communicator using shared memory for intra-node communication
/// @details Provides efficient point-to-point and collective operations
///          for processes/threads on the same node using shared memory.
///          Uses POSIX shm_open/mmap on Linux/macOS, with heap fallback
///          for single-process testing.
///
/// @par Status: Functional / single-node only
/// @par Limitations:
///     - Single mailbox per sender-receiver pair (no concurrent sends to same dest)
///     - O(p) work at rank 0 for allreduce
///     - No tree-based collectives
class shared_memory_communicator : public communicator_base {
public:
    /// @brief Configuration for shared memory communicator
    struct config {
        /// @brief Default constructor
        config() : mailbox_size(64 * 1024),
                   collective_buffer_size(1024 * 1024),
                   busy_wait(true) {}

        /// @brief Size of each mailbox for point-to-point messages
        size_type mailbox_size;  // 64 KB

        /// @brief Size of collective buffer
        size_type collective_buffer_size;  // 1 MB

        /// @brief Whether to use busy-waiting (vs. futex/sleep)
        bool busy_wait;
    };

    /// @brief Default constructor
    shared_memory_communicator() = default;

    /// @brief Construct communicator for given ranks
    /// @param rank This process's rank
    /// @param size Total number of ranks
    /// @param cfg Configuration options
    explicit shared_memory_communicator(rank_t rank, rank_t size,
                                        const config& cfg = config())
        : rank_(rank)
        , size_(size)
        , config_(cfg) {
        initialize();
    }

    /// @brief Destructor
    ~shared_memory_communicator() override = default;

    // Non-copyable
    shared_memory_communicator(const shared_memory_communicator&) = delete;
    shared_memory_communicator& operator=(const shared_memory_communicator&) = delete;

    // Movable
    shared_memory_communicator(shared_memory_communicator&&) = default;
    shared_memory_communicator& operator=(shared_memory_communicator&&) = default;

    // ------------------------------------------------------------------------
    // Communicator Interface
    // ------------------------------------------------------------------------

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept override { return rank_; }

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept override { return size_; }

    /// @brief Check if communicator is valid
    [[nodiscard]] bool valid() const noexcept {
        return size_ > 0 && rank_ >= 0 && rank_ < size_;
    }

    /// @brief Get communicator properties
    [[nodiscard]] communicator_properties properties() const noexcept override {
        return communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "shared_memory"
        };
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Communication
    // ------------------------------------------------------------------------

    /// @brief Blocking send via shared memory mailbox
    /// @details Writes tag, size, and payload into the destination rank's
    ///          mailbox, then sets the flag to signal the receiver.
    result<void> send_impl(const void* data, size_type count,
                          size_type elem_size, rank_t dest, int tag) {
        if (dest < 0 || dest >= size_) {
            return make_error<void>(status_code::invalid_argument,
                                    "Invalid destination rank");
        }

        size_type total_size = count * elem_size;
        size_type usable = config_.mailbox_size - sizeof(mailbox_header);
        if (total_size > usable) {
            return make_error<void>(status_code::buffer_too_small,
                                    "Message exceeds mailbox capacity");
        }

        // Get the mailbox for (sender=rank_, receiver=dest)
        auto* hdr = get_mailbox(rank_, dest);
        if (!hdr) {
            return make_error<void>(status_code::invalid_state,
                                    "Mailbox not initialized");
        }

        // Write header fields
        hdr->tag = tag;
        hdr->msg_size = total_size;

        // Write payload after header
        if (total_size > 0) {
            auto* payload = reinterpret_cast<char*>(hdr) + sizeof(mailbox_header);
            std::memcpy(payload, data, total_size);
        }

        // Signal the receiver (release ensures payload is visible)
        hdr->flag.store(1, std::memory_order_release);
        hdr->flag.notify_all();

        return {};
    }

    /// @brief Blocking receive via shared memory mailbox
    /// @details Waits for the flag in the sender's mailbox to be set,
    ///          then copies data from the mailbox and resets the flag.
    result<void> recv_impl(void* data, size_type count,
                          size_type elem_size, rank_t source, int tag) {
        if (source < 0 || source >= size_) {
            return make_error<void>(status_code::invalid_argument,
                                    "Invalid source rank");
        }

        size_type total_size = count * elem_size;

        // Get the mailbox for (sender=source, receiver=rank_)
        auto* hdr = get_mailbox(source, rank_);
        if (!hdr) {
            return make_error<void>(status_code::invalid_state,
                                    "Mailbox not initialized");
        }

        // Wait for message (C++20 atomic wait)
        hdr->flag.wait(0, std::memory_order_acquire);

        // Tag validation is advisory
        (void)tag;

        // Copy payload
        size_type copy_size = (total_size < hdr->msg_size) ? total_size : hdr->msg_size;
        if (copy_size > 0) {
            auto* payload = reinterpret_cast<const char*>(hdr) + sizeof(mailbox_header);
            std::memcpy(data, payload, copy_size);
        }

        // Reset flag (ready for next message)
        hdr->flag.store(0, std::memory_order_release);

        return {};
    }

    // ------------------------------------------------------------------------
    // Collective Communication
    // ------------------------------------------------------------------------

    /// @brief Barrier synchronization
    result<void> barrier() {
        if (!barrier_) {
            return make_error<void>(status_code::invalid_state,
                                    "Communicator not initialized");
        }
        barrier_->arrive_and_wait();
        return {};
    }

    /// @brief Broadcast from root to all ranks
    /// @details Root writes to collective buffer, barrier, non-root copies out.
    result<void> broadcast_impl(void* data, size_type count,
                               size_type elem_size, rank_t root) {
        if (root < 0 || root >= size_) {
            return make_error<void>(status_code::invalid_argument,
                                    "Invalid root rank");
        }

        size_type total_size = count * elem_size;

        // Root copies to shared buffer
        if (rank_ == root) {
            if (collective_region_ && total_size <= collective_region_->size()) {
                std::memcpy(collective_region_->data(), data, total_size);
            }
        }

        // Barrier
        if (barrier_) barrier_->arrive_and_wait();

        // Non-root copies from shared buffer
        if (rank_ != root) {
            if (collective_region_ && total_size <= collective_region_->size()) {
                std::memcpy(data, collective_region_->data(), total_size);
            }
        }

        // Final barrier to ensure root doesn't overwrite before all copy
        if (barrier_) barrier_->arrive_and_wait();

        return {};
    }

    /// @brief Gather data from all ranks to root
    /// @details Each rank writes its data to its slot in the collective buffer,
    ///          barrier, root copies all slots to recv_data.
    /// @note O(p) work at root; tree-based gather would be more scalable.
    result<void> gather_impl(const void* send_data, size_type send_count,
                            void* recv_data, size_type recv_count,
                            size_type elem_size, rank_t root) {
        if (root < 0 || root >= size_) {
            return make_error<void>(status_code::invalid_argument,
                                    "Invalid root rank");
        }

        size_type send_bytes = send_count * elem_size;
        size_type total_needed = static_cast<size_type>(size_) * send_bytes;

        if (!collective_region_ || total_needed > collective_region_->size()) {
            return make_error<void>(status_code::buffer_too_small,
                                    "Collective buffer too small for gather");
        }

        // Each rank writes to its slot
        auto* buf = static_cast<char*>(collective_region_->data());
        std::memcpy(buf + static_cast<size_type>(rank_) * send_bytes,
                    send_data, send_bytes);

        // Barrier: all ranks have written
        if (barrier_) barrier_->arrive_and_wait();

        // Root copies all slots to recv_data
        if (rank_ == root) {
            size_type recv_bytes = recv_count * elem_size;
            (void)recv_bytes;  // recv_count should == send_count per rank
            std::memcpy(recv_data, buf, total_needed);
        }

        // Final barrier: safe to reuse collective buffer
        if (barrier_) barrier_->arrive_and_wait();

        return {};
    }

    /// @brief Scatter data from root to all ranks
    /// @details Root writes to collective buffer, barrier, each rank copies its slot.
    result<void> scatter_impl(const void* send_data, size_type send_count,
                             void* recv_data, size_type recv_count,
                             size_type elem_size, rank_t root) {
        if (root < 0 || root >= size_) {
            return make_error<void>(status_code::invalid_argument,
                                    "Invalid root rank");
        }

        size_type recv_bytes = recv_count * elem_size;
        size_type total_needed = static_cast<size_type>(size_) * send_count * elem_size;

        if (!collective_region_ || total_needed > collective_region_->size()) {
            return make_error<void>(status_code::buffer_too_small,
                                    "Collective buffer too small for scatter");
        }

        auto* buf = static_cast<char*>(collective_region_->data());

        // Root writes all data to collective buffer
        if (rank_ == root) {
            std::memcpy(buf, send_data, total_needed);
        }

        // Barrier: root has written
        if (barrier_) barrier_->arrive_and_wait();

        // Each rank copies its slot
        size_type send_bytes = send_count * elem_size;
        std::memcpy(recv_data, buf + static_cast<size_type>(rank_) * send_bytes,
                    recv_bytes);

        // Final barrier: safe to reuse collective buffer
        if (barrier_) barrier_->arrive_and_wait();

        return {};
    }

    /// @brief Allgather: gather + broadcast
    /// @details Each rank writes to its slot, barrier, all ranks copy the
    ///          entire buffer.
    result<void> allgather_impl(const void* send_data, size_type send_count,
                               void* recv_data, size_type recv_count,
                               size_type elem_size) {
        size_type send_bytes = send_count * elem_size;
        size_type total_needed = static_cast<size_type>(size_) * send_bytes;

        if (!collective_region_ || total_needed > collective_region_->size()) {
            return make_error<void>(status_code::buffer_too_small,
                                    "Collective buffer too small for allgather");
        }

        auto* buf = static_cast<char*>(collective_region_->data());

        // Each rank writes to its slot
        std::memcpy(buf + static_cast<size_type>(rank_) * send_bytes,
                    send_data, send_bytes);

        // Barrier: all have written
        if (barrier_) barrier_->arrive_and_wait();

        // All ranks copy the full buffer
        (void)recv_count;
        std::memcpy(recv_data, buf, total_needed);

        // Final barrier
        if (barrier_) barrier_->arrive_and_wait();

        return {};
    }

    // ------------------------------------------------------------------------
    // Shared Memory Specific
    // ------------------------------------------------------------------------

    /// @brief Get pointer to shared region for direct access
    /// @return Pointer to collective shared region
    [[nodiscard]] void* shared_buffer() noexcept {
        return collective_region_ ? collective_region_->data() : nullptr;
    }

    /// @brief Get size of shared buffer
    [[nodiscard]] size_type shared_buffer_size() const noexcept {
        return collective_region_ ? collective_region_->size() : 0;
    }

private:
    /// @brief Get the mailbox header for a (sender, receiver) pair
    /// @details Mailboxes are indexed as mailbox_regions_[sender * size_ + receiver].
    ///          Each mailbox is config_.mailbox_size bytes, laid out as:
    ///          [mailbox_header] [payload data]
    ///          Returns nullptr if the entry is null or the region's data is null.
    mailbox_header* get_mailbox(rank_t sender, rank_t receiver) {
        size_type idx = static_cast<size_type>(sender) * static_cast<size_type>(size_)
                      + static_cast<size_type>(receiver);
        if (idx >= mailbox_regions_.size() || !mailbox_regions_[idx]) {
            return nullptr;
        }
        void* data = mailbox_regions_[idx]->data();
        if (!data) {
            return nullptr;
        }
        return reinterpret_cast<mailbox_header*>(data);
    }

    void initialize() {
        // Create collective shared region
        collective_region_ = std::make_unique<shared_region>(
            config_.collective_buffer_size, 0);

        // Create per-pair mailbox regions
        // Layout: one mailbox per (sender, receiver) pair
        size_type num_pairs = static_cast<size_type>(size_) * static_cast<size_type>(size_);
        mailbox_regions_.reserve(num_pairs);
        for (size_type i = 0; i < num_pairs; ++i) {
            auto region = std::make_unique<shared_region>(
                config_.mailbox_size, 1000 + i);
            if (region->valid()) {
                // Zero-initialize (flag = 0 means empty)
                std::memset(region->data(), 0, config_.mailbox_size);
                mailbox_regions_.push_back(std::move(region));
            } else {
                // Push nullptr for invalid regions so get_mailbox returns nullptr
                mailbox_regions_.push_back(nullptr);
            }
        }

        // Create barrier
        barrier_ = std::make_unique<shared_barrier>(size_);
    }

    rank_t rank_ = no_rank;
    rank_t size_ = 0;
    config config_;

    std::unique_ptr<shared_region> collective_region_;
    std::vector<std::unique_ptr<shared_region>> mailbox_regions_;
    std::unique_ptr<shared_barrier> barrier_;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Create a shared memory communicator
/// @param rank This process's rank
/// @param size Total number of ranks
/// @return Shared memory communicator
[[nodiscard]] inline std::unique_ptr<shared_memory_communicator>
make_shared_memory_communicator(rank_t rank, rank_t size) {
    return std::make_unique<shared_memory_communicator>(rank, size);
}

}  // namespace shared_memory
}  // namespace dtl
