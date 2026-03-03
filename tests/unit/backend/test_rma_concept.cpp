// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rma_concept.cpp
/// @brief Unit tests for RmaCommunicator concept
/// @details Verifies RMA concept requirements using mock implementations.

#include <dtl/backend/concepts/rma_communicator.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Mock Communicator Base (satisfies Communicator concept)
// =============================================================================

/// @brief Minimal mock communicator that satisfies the Communicator concept
struct mock_communicator_base {
    using size_type = dtl::size_type;

    [[nodiscard]] rank_t rank() const { return rank_; }
    [[nodiscard]] rank_t size() const { return size_; }

    void send(const void*, size_type, rank_t, int) {}
    void recv(void*, size_type, rank_t, int) {}
    [[nodiscard]] request_handle isend(const void*, size_type, rank_t, int) {
        return request_handle{reinterpret_cast<void*>(1)};
    }
    [[nodiscard]] request_handle irecv(void*, size_type, rank_t, int) {
        return request_handle{reinterpret_cast<void*>(2)};
    }
    void wait(request_handle& req) { req.handle = nullptr; }
    [[nodiscard]] bool test(request_handle& req) {
        if (req.handle != nullptr) {
            req.handle = nullptr;
            return true;
        }
        return false;
    }

private:
    rank_t rank_ = 0;
    rank_t size_ = 1;
};

// =============================================================================
// Mock RMA Communicator (satisfies RmaCommunicator concept)
// =============================================================================

/// @brief Mock RMA communicator that satisfies the RmaCommunicator concept
struct mock_rma_communicator : mock_communicator_base {
    // Window management
    [[nodiscard]] window_handle create_window(void* base, size_type size) {
        last_window_base_ = base;
        last_window_size_ = size;
        return window_handle{reinterpret_cast<void*>(0x1000)};
    }

    void free_window(window_handle& win) {
        win.handle = nullptr;
    }

    // One-sided data transfer
    void put(const void* origin, size_type size, rank_t target,
             size_type offset, window_handle& /*win*/) {
        last_put_origin_ = origin;
        last_put_size_ = size;
        last_put_target_ = target;
        last_put_offset_ = offset;
    }

    void get(void* origin, size_type size, rank_t target,
             size_type offset, window_handle& /*win*/) {
        last_get_origin_ = origin;
        last_get_size_ = size;
        last_get_target_ = target;
        last_get_offset_ = offset;
    }

    // Active-target synchronization
    void fence(window_handle& /*win*/) {
        fence_count_++;
    }

    // Remote completion
    void flush(rank_t target, window_handle& /*win*/) {
        last_flush_target_ = target;
        flush_count_++;
    }

    void flush_all(window_handle& /*win*/) {
        flush_all_count_++;
    }

    // Test inspection
    void* last_window_base_ = nullptr;
    size_type last_window_size_ = 0;
    const void* last_put_origin_ = nullptr;
    size_type last_put_size_ = 0;
    rank_t last_put_target_ = -1;
    size_type last_put_offset_ = 0;
    void* last_get_origin_ = nullptr;
    size_type last_get_size_ = 0;
    rank_t last_get_target_ = -1;
    size_type last_get_offset_ = 0;
    int fence_count_ = 0;
    rank_t last_flush_target_ = -1;
    int flush_count_ = 0;
    int flush_all_count_ = 0;
};

// =============================================================================
// Mock Passive Target RMA Communicator
// =============================================================================

/// @brief Mock RMA communicator with passive-target support
struct mock_passive_rma_communicator : mock_rma_communicator {
    void lock(rank_t target, rma_lock_mode mode, window_handle& /*win*/) {
        last_lock_target_ = target;
        last_lock_mode_ = mode;
        lock_count_++;
    }

    void unlock(rank_t target, window_handle& /*win*/) {
        last_unlock_target_ = target;
        unlock_count_++;
    }

    void lock_all(window_handle& /*win*/) {
        lock_all_count_++;
    }

    void unlock_all(window_handle& /*win*/) {
        unlock_all_count_++;
    }

    rank_t last_lock_target_ = -1;
    rma_lock_mode last_lock_mode_ = rma_lock_mode::exclusive;
    int lock_count_ = 0;
    rank_t last_unlock_target_ = -1;
    int unlock_count_ = 0;
    int lock_all_count_ = 0;
    int unlock_all_count_ = 0;
};

// =============================================================================
// Mock Atomic RMA Communicator
// =============================================================================

/// @brief Mock RMA communicator with atomic operations
struct mock_atomic_rma_communicator : mock_rma_communicator {
    void accumulate(const void* origin, size_type size, rank_t target,
                   size_type offset, rma_reduce_op op, window_handle& /*win*/) {
        last_acc_origin_ = origin;
        last_acc_size_ = size;
        last_acc_target_ = target;
        last_acc_offset_ = offset;
        last_acc_op_ = op;
        accumulate_count_++;
    }

    void fetch_and_op(const void* origin, void* result, size_type size,
                      rank_t target, size_type offset, rma_reduce_op op,
                      window_handle& /*win*/) {
        last_fao_origin_ = origin;
        last_fao_result_ = result;
        last_fao_size_ = size;
        last_fao_target_ = target;
        last_fao_offset_ = offset;
        last_fao_op_ = op;
        fetch_and_op_count_++;
    }

    void compare_and_swap(const void* compare, const void* swap_val, void* result,
                          size_type size, rank_t target, size_type offset,
                          window_handle& /*win*/) {
        last_cas_compare_ = compare;
        last_cas_swap_ = swap_val;
        last_cas_result_ = result;
        last_cas_size_ = size;
        last_cas_target_ = target;
        last_cas_offset_ = offset;
        cas_count_++;
    }

    const void* last_acc_origin_ = nullptr;
    size_type last_acc_size_ = 0;
    rank_t last_acc_target_ = -1;
    size_type last_acc_offset_ = 0;
    rma_reduce_op last_acc_op_ = rma_reduce_op::sum;
    int accumulate_count_ = 0;

    const void* last_fao_origin_ = nullptr;
    void* last_fao_result_ = nullptr;
    size_type last_fao_size_ = 0;
    rank_t last_fao_target_ = -1;
    size_type last_fao_offset_ = 0;
    rma_reduce_op last_fao_op_ = rma_reduce_op::sum;
    int fetch_and_op_count_ = 0;

    const void* last_cas_compare_ = nullptr;
    const void* last_cas_swap_ = nullptr;
    void* last_cas_result_ = nullptr;
    size_type last_cas_size_ = 0;
    rank_t last_cas_target_ = -1;
    size_type last_cas_offset_ = 0;
    int cas_count_ = 0;
};

// =============================================================================
// Mock Full RMA Communicator
// =============================================================================

/// @brief Mock RMA communicator with all capabilities
struct mock_full_rma_communicator : mock_passive_rma_communicator {
    void accumulate(const void* origin, size_type size, rank_t target,
                   size_type offset, rma_reduce_op op, window_handle& /*win*/) {
        last_acc_op_ = op;
        accumulate_count_++;
        (void)origin; (void)size; (void)target; (void)offset;
    }

    void fetch_and_op(const void* origin, void* result, size_type size,
                      rank_t target, size_type offset, rma_reduce_op op,
                      window_handle& /*win*/) {
        last_fao_op_ = op;
        fetch_and_op_count_++;
        (void)origin; (void)result; (void)size; (void)target; (void)offset;
    }

    void compare_and_swap(const void* compare, const void* swap_val, void* result,
                          size_type size, rank_t target, size_type offset,
                          window_handle& /*win*/) {
        cas_count_++;
        (void)compare; (void)swap_val; (void)result; (void)size; (void)target; (void)offset;
    }

    rma_reduce_op last_acc_op_ = rma_reduce_op::sum;
    int accumulate_count_ = 0;
    rma_reduce_op last_fao_op_ = rma_reduce_op::sum;
    int fetch_and_op_count_ = 0;
    int cas_count_ = 0;
};

// =============================================================================
// Non-Conforming Types
// =============================================================================

/// @brief Type that satisfies Communicator but not RmaCommunicator
struct not_rma_communicator : mock_communicator_base {
    // Missing: create_window, free_window, put, get, fence, flush, flush_all
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(RmaConceptTest, MockRmaCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_rma_communicator>);
    static_assert(RmaCommunicator<mock_rma_communicator>);
}

TEST(RmaConceptTest, MockPassiveRmaCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_passive_rma_communicator>);
    static_assert(RmaCommunicator<mock_passive_rma_communicator>);
    static_assert(PassiveTargetRmaCommunicator<mock_passive_rma_communicator>);
}

TEST(RmaConceptTest, MockAtomicRmaCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_atomic_rma_communicator>);
    static_assert(RmaCommunicator<mock_atomic_rma_communicator>);
    static_assert(AtomicRmaCommunicator<mock_atomic_rma_communicator>);
}

TEST(RmaConceptTest, MockFullRmaCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_full_rma_communicator>);
    static_assert(RmaCommunicator<mock_full_rma_communicator>);
    static_assert(PassiveTargetRmaCommunicator<mock_full_rma_communicator>);
    static_assert(AtomicRmaCommunicator<mock_full_rma_communicator>);
    static_assert(FullRmaCommunicator<mock_full_rma_communicator>);
}

TEST(RmaConceptTest, NonRmaCommunicatorDoesNotSatisfyConcept) {
    static_assert(Communicator<not_rma_communicator>);
    static_assert(!RmaCommunicator<not_rma_communicator>);
}

TEST(RmaConceptTest, BasicTypesDontSatisfyConcept) {
    static_assert(!RmaCommunicator<int>);
    static_assert(!RmaCommunicator<double>);
    static_assert(!RmaCommunicator<std::string>);
}

TEST(RmaConceptTest, BasicRmaDoesNotSatisfyPassiveTarget) {
    static_assert(RmaCommunicator<mock_rma_communicator>);
    static_assert(!PassiveTargetRmaCommunicator<mock_rma_communicator>);
}

TEST(RmaConceptTest, BasicRmaDoesNotSatisfyAtomic) {
    static_assert(RmaCommunicator<mock_rma_communicator>);
    static_assert(!AtomicRmaCommunicator<mock_rma_communicator>);
}

// =============================================================================
// Window Handle Tests
// =============================================================================

TEST(WindowHandleTest, DefaultConstruction) {
    window_handle win;
    EXPECT_FALSE(win.valid());
    EXPECT_EQ(win.handle, nullptr);
}

TEST(WindowHandleTest, ValidCheck) {
    window_handle win{reinterpret_cast<void*>(0x1000)};
    EXPECT_TRUE(win.valid());
}

TEST(WindowHandleTest, NullptrIsInvalid) {
    window_handle win{nullptr};
    EXPECT_FALSE(win.valid());
}

TEST(WindowHandleTest, EqualityComparison) {
    window_handle win1{reinterpret_cast<void*>(0x1000)};
    window_handle win2{reinterpret_cast<void*>(0x1000)};
    window_handle win3{reinterpret_cast<void*>(0x2000)};
    window_handle win4{};

    EXPECT_EQ(win1, win2);
    EXPECT_NE(win1, win3);
    EXPECT_NE(win1, win4);
}

// =============================================================================
// RMA Lock Mode Tests
// =============================================================================

TEST(RmaLockModeTest, EnumValues) {
    EXPECT_NE(static_cast<int>(rma_lock_mode::exclusive),
              static_cast<int>(rma_lock_mode::shared));
}

TEST(RmaLockModeTest, CanUseInSwitch) {
    rma_lock_mode mode = rma_lock_mode::exclusive;
    bool is_exclusive = false;

    switch (mode) {
        case rma_lock_mode::exclusive:
            is_exclusive = true;
            break;
        case rma_lock_mode::shared:
            is_exclusive = false;
            break;
    }

    EXPECT_TRUE(is_exclusive);
}

// =============================================================================
// RMA Reduce Op Tests
// =============================================================================

TEST(RmaReduceOpTest, AllOpsAreDifferent) {
    EXPECT_NE(static_cast<int>(rma_reduce_op::sum), static_cast<int>(rma_reduce_op::prod));
    EXPECT_NE(static_cast<int>(rma_reduce_op::min), static_cast<int>(rma_reduce_op::max));
    EXPECT_NE(static_cast<int>(rma_reduce_op::band), static_cast<int>(rma_reduce_op::bor));
    EXPECT_NE(static_cast<int>(rma_reduce_op::bxor), static_cast<int>(rma_reduce_op::replace));
    EXPECT_NE(static_cast<int>(rma_reduce_op::no_op), static_cast<int>(rma_reduce_op::sum));
}

TEST(RmaReduceOpTest, CanUseInSwitch) {
    rma_reduce_op op = rma_reduce_op::sum;
    bool is_sum = false;

    switch (op) {
        case rma_reduce_op::sum:      is_sum = true; break;
        case rma_reduce_op::prod:     break;
        case rma_reduce_op::min:      break;
        case rma_reduce_op::max:      break;
        case rma_reduce_op::band:     break;
        case rma_reduce_op::bor:      break;
        case rma_reduce_op::bxor:     break;
        case rma_reduce_op::replace:  break;
        case rma_reduce_op::no_op:    break;
    }

    EXPECT_TRUE(is_sum);
}

// =============================================================================
// Capability Trait Tests
// =============================================================================

TEST(RmaCapabilityTraitTest, SupportsRmaV) {
    static_assert(supports_rma_v<mock_rma_communicator>);
    static_assert(!supports_rma_v<not_rma_communicator>);
    static_assert(!supports_rma_v<int>);
}

TEST(RmaCapabilityTraitTest, SupportsPassiveTargetRmaV) {
    static_assert(supports_passive_target_rma_v<mock_passive_rma_communicator>);
    static_assert(!supports_passive_target_rma_v<mock_rma_communicator>);
    static_assert(!supports_passive_target_rma_v<int>);
}

TEST(RmaCapabilityTraitTest, SupportsAtomicRmaV) {
    static_assert(supports_atomic_rma_v<mock_atomic_rma_communicator>);
    static_assert(!supports_atomic_rma_v<mock_rma_communicator>);
    static_assert(!supports_atomic_rma_v<int>);
}

TEST(RmaCapabilityTraitTest, SupportsFullRmaV) {
    static_assert(supports_full_rma_v<mock_full_rma_communicator>);
    static_assert(!supports_full_rma_v<mock_passive_rma_communicator>);
    static_assert(!supports_full_rma_v<mock_atomic_rma_communicator>);
    static_assert(!supports_full_rma_v<int>);
}

// =============================================================================
// Mock Operation Tests
// =============================================================================

TEST(RmaMockTest, WindowCreation) {
    mock_rma_communicator comm;
    int data = 42;

    auto win = comm.create_window(&data, sizeof(data));
    EXPECT_TRUE(win.valid());
    EXPECT_EQ(comm.last_window_base_, &data);
    EXPECT_EQ(comm.last_window_size_, sizeof(data));
}

TEST(RmaMockTest, WindowFree) {
    mock_rma_communicator comm;
    int data = 42;

    auto win = comm.create_window(&data, sizeof(data));
    EXPECT_TRUE(win.valid());

    comm.free_window(win);
    EXPECT_FALSE(win.valid());
}

TEST(RmaMockTest, PutOperation) {
    mock_rma_communicator comm;
    int local_data = 42;
    int window_data = 0;

    auto win = comm.create_window(&window_data, sizeof(window_data));
    comm.put(&local_data, sizeof(local_data), 1, 0, win);

    EXPECT_EQ(comm.last_put_origin_, &local_data);
    EXPECT_EQ(comm.last_put_size_, sizeof(local_data));
    EXPECT_EQ(comm.last_put_target_, 1);
    EXPECT_EQ(comm.last_put_offset_, 0);
}

TEST(RmaMockTest, GetOperation) {
    mock_rma_communicator comm;
    int local_buffer = 0;
    int window_data = 42;

    auto win = comm.create_window(&window_data, sizeof(window_data));
    comm.get(&local_buffer, sizeof(local_buffer), 1, 8, win);

    EXPECT_EQ(comm.last_get_origin_, &local_buffer);
    EXPECT_EQ(comm.last_get_size_, sizeof(local_buffer));
    EXPECT_EQ(comm.last_get_target_, 1);
    EXPECT_EQ(comm.last_get_offset_, 8);
}

TEST(RmaMockTest, FenceOperation) {
    mock_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));

    EXPECT_EQ(comm.fence_count_, 0);
    comm.fence(win);
    EXPECT_EQ(comm.fence_count_, 1);
    comm.fence(win);
    EXPECT_EQ(comm.fence_count_, 2);
}

TEST(RmaMockTest, FlushOperation) {
    mock_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));

    EXPECT_EQ(comm.flush_count_, 0);
    comm.flush(2, win);
    EXPECT_EQ(comm.flush_count_, 1);
    EXPECT_EQ(comm.last_flush_target_, 2);
}

TEST(RmaMockTest, FlushAllOperation) {
    mock_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));

    EXPECT_EQ(comm.flush_all_count_, 0);
    comm.flush_all(win);
    EXPECT_EQ(comm.flush_all_count_, 1);
}

TEST(RmaMockTest, LockUnlockOperations) {
    mock_passive_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));

    EXPECT_EQ(comm.lock_count_, 0);
    comm.lock(1, rma_lock_mode::exclusive, win);
    EXPECT_EQ(comm.lock_count_, 1);
    EXPECT_EQ(comm.last_lock_target_, 1);
    EXPECT_EQ(comm.last_lock_mode_, rma_lock_mode::exclusive);

    comm.unlock(1, win);
    EXPECT_EQ(comm.unlock_count_, 1);
    EXPECT_EQ(comm.last_unlock_target_, 1);
}

TEST(RmaMockTest, LockAllUnlockAllOperations) {
    mock_passive_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));

    EXPECT_EQ(comm.lock_all_count_, 0);
    comm.lock_all(win);
    EXPECT_EQ(comm.lock_all_count_, 1);

    comm.unlock_all(win);
    EXPECT_EQ(comm.unlock_all_count_, 1);
}

TEST(RmaMockTest, AccumulateOperation) {
    mock_atomic_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));
    int value = 10;

    comm.accumulate(&value, sizeof(value), 1, 0, rma_reduce_op::sum, win);
    EXPECT_EQ(comm.accumulate_count_, 1);
    EXPECT_EQ(comm.last_acc_origin_, &value);
    EXPECT_EQ(comm.last_acc_op_, rma_reduce_op::sum);
}

TEST(RmaMockTest, FetchAndOpOperation) {
    mock_atomic_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));
    int value = 10;
    int result = 0;

    comm.fetch_and_op(&value, &result, sizeof(value), 1, 0, rma_reduce_op::max, win);
    EXPECT_EQ(comm.fetch_and_op_count_, 1);
    EXPECT_EQ(comm.last_fao_origin_, &value);
    EXPECT_EQ(comm.last_fao_result_, &result);
    EXPECT_EQ(comm.last_fao_op_, rma_reduce_op::max);
}

TEST(RmaMockTest, CompareAndSwapOperation) {
    mock_atomic_rma_communicator comm;
    int data = 42;
    auto win = comm.create_window(&data, sizeof(data));
    int compare = 42;
    int swap = 100;
    int result = 0;

    comm.compare_and_swap(&compare, &swap, &result, sizeof(int), 1, 0, win);
    EXPECT_EQ(comm.cas_count_, 1);
    EXPECT_EQ(comm.last_cas_compare_, &compare);
    EXPECT_EQ(comm.last_cas_swap_, &swap);
    EXPECT_EQ(comm.last_cas_result_, &result);
}

// =============================================================================
// Tag Type Tests
// =============================================================================

TEST(RmaTagTest, TagIsDistinct) {
    static_assert(!std::is_same_v<rma_communicator_tag, mpi_communicator_tag>);
    static_assert(!std::is_same_v<rma_communicator_tag, shared_memory_communicator_tag>);
    static_assert(!std::is_same_v<rma_communicator_tag, gpu_communicator_tag>);
}

}  // namespace dtl::test
