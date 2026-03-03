// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_rma.cpp
/// @brief Unit tests for MPI RMA backend
/// @details Tests mpi_window and mpi_rma_adapter implementations.

#include <backends/mpi/mpi_window.hpp>
#include <backends/mpi/mpi_rma_adapter.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>

#include <gtest/gtest.h>
#include <array>
#include <vector>

namespace dtl::test {

// =============================================================================
// MPI Window Tests
// =============================================================================

class MpiWindowTest : public ::testing::Test {
protected:
    std::array<int, 100> data_{};
};

TEST_F(MpiWindowTest, DefaultConstructorCreatesInvalidWindow) {
    mpi::mpi_window window;
    EXPECT_FALSE(window.valid());
    EXPECT_EQ(window.base(), nullptr);
    EXPECT_EQ(window.size(), 0u);
}

TEST_F(MpiWindowTest, MoveConstructorTransfersOwnership) {
    mpi::mpi_window window1;
    // Can't test with real MPI window without MPI, but can test move semantics
    mpi::mpi_window window2(std::move(window1));
    EXPECT_FALSE(window1.valid());
    EXPECT_FALSE(window2.valid());  // Both invalid since no real window created
}

TEST_F(MpiWindowTest, MoveAssignmentTransfersOwnership) {
    mpi::mpi_window window1;
    mpi::mpi_window window2;
    window2 = std::move(window1);
    EXPECT_FALSE(window1.valid());
    EXPECT_FALSE(window2.valid());
}

TEST_F(MpiWindowTest, InvalidWindowPutFails) {
    mpi::mpi_window window;
    int value = 42;
    auto result = window.put_impl(&value, sizeof(int), 0, 0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowGetFails) {
    mpi::mpi_window window;
    int value = 0;
    auto result = window.get_impl(&value, sizeof(int), 0, 0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFenceFails) {
    mpi::mpi_window window;
    auto result = window.fence_impl();
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowLockFails) {
    mpi::mpi_window window;
#if DTL_ENABLE_MPI
    auto result = window.lock_impl(0, MPI_LOCK_EXCLUSIVE);
#else
    auto result = window.lock_impl(0, 0);
#endif
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowUnlockFails) {
    mpi::mpi_window window;
    auto result = window.unlock_impl(0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowLockAllFails) {
    mpi::mpi_window window;
    auto result = window.lock_all_impl();
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowUnlockAllFails) {
    mpi::mpi_window window;
    auto result = window.unlock_all_impl();
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFlushFails) {
    mpi::mpi_window window;
    auto result = window.flush_impl(0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFlushAllFails) {
    mpi::mpi_window window;
    auto result = window.flush_all_impl();
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFlushLocalFails) {
    mpi::mpi_window window;
    auto result = window.flush_local_impl(0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFlushLocalAllFails) {
    mpi::mpi_window window;
    auto result = window.flush_local_all_impl();
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowAccumulateFails) {
    mpi::mpi_window window;
    int value = 42;
    auto result = window.accumulate_impl(&value, sizeof(int), 0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowFetchAndOpFails) {
    mpi::mpi_window window;
    int origin = 42;
    int result_val = 0;
    auto result = window.fetch_and_op_impl(&origin, &result_val, sizeof(int),
                                            0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowCompareAndSwapFails) {
    mpi::mpi_window window;
    int origin = 42;
    int compare = 0;
    int result_val = 0;
    auto result = window.compare_and_swap_impl(&origin, &compare, &result_val,
                                                sizeof(int), 0, 0);
    EXPECT_FALSE(result.has_value());
}

TEST_F(MpiWindowTest, InvalidWindowGetAccumulateFails) {
    mpi::mpi_window window;
    int origin = 42;
    int result_val = 0;
    auto result = window.get_accumulate_impl(&origin, &result_val, sizeof(int),
                                              0, 0, rma_reduce_op::sum);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// MPI RMA Adapter Tests
// =============================================================================

class MpiRmaAdapterTest : public ::testing::Test {
protected:
    std::array<int, 100> data_{};
};

TEST_F(MpiRmaAdapterTest, DefaultConstruction) {
    mpi::mpi_rma_adapter adapter;
    // Without MPI init, rank/size will be default values
    // Just verify it constructs
    (void)adapter.rank();
    (void)adapter.size();
}

TEST_F(MpiRmaAdapterTest, CommAccessor) {
    mpi::mpi_rma_adapter adapter;
    auto& comm = adapter.comm();
    (void)comm.rank();  // Just verify we can access it
}

TEST_F(MpiRmaAdapterTest, ConstCommAccessor) {
    const mpi::mpi_rma_adapter adapter;
    const auto& comm = adapter.comm();
    (void)comm.rank();
}

TEST_F(MpiRmaAdapterTest, FreeNullWindowNoThrow) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;  // Invalid handle
    EXPECT_NO_THROW(adapter.free_window(win));
}

TEST_F(MpiRmaAdapterTest, PutWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;  // Invalid handle
    int value = 42;
    EXPECT_THROW(adapter.put(&value, sizeof(int), 0, 0, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, GetWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;  // Invalid handle
    int value = 0;
    EXPECT_THROW(adapter.get(&value, sizeof(int), 0, 0, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, FenceWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.fence(win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, FlushWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.flush(0, win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, FlushAllWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.flush_all(win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, LockWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.lock(0, rma_lock_mode::exclusive, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, UnlockWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.unlock(0, win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, LockAllWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.lock_all(win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, UnlockAllWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    EXPECT_THROW(adapter.unlock_all(win), mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, AccumulateWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    int value = 42;
    EXPECT_THROW(adapter.accumulate(&value, sizeof(int), 0, 0,
                                    rma_reduce_op::sum, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, FetchAndOpWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    int origin = 42;
    int result_val = 0;
    EXPECT_THROW(adapter.fetch_and_op(&origin, &result_val, sizeof(int),
                                       0, 0, rma_reduce_op::sum, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, CompareAndSwapWithInvalidWindowThrows) {
    mpi::mpi_rma_adapter adapter;
    window_handle win;
    int origin = 42;
    int compare = 0;
    int result_val = 0;
    EXPECT_THROW(adapter.compare_and_swap(&origin, &compare, &result_val,
                                           sizeof(int), 0, 0, win),
                 mpi::communication_error);
}

TEST_F(MpiRmaAdapterTest, WorldRmaAdapterFactory) {
    auto adapter = mpi::world_rma_adapter();
    (void)adapter.rank();
}

// =============================================================================
// Reduce Op Mapping Tests (for MPI)
// =============================================================================

TEST(MpiReduceOpTest, AllReduceOpsHaveValidValues) {
    // Verify all enum values are defined and usable
    EXPECT_NE(static_cast<int>(rma_reduce_op::sum), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::prod), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::min), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::max), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::band), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::bor), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::bxor), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::replace), -1);
    EXPECT_NE(static_cast<int>(rma_reduce_op::no_op), -1);
}

// =============================================================================
// Concept Satisfaction Tests
// =============================================================================

#if DTL_ENABLE_MPI
TEST(ConceptSatisfactionTest, MpiRmaAdapterSatisfiesRmaCommunicator) {
    static_assert(RmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy RmaCommunicator");
}

TEST(ConceptSatisfactionTest, MpiRmaAdapterSatisfiesPassiveTargetRmaCommunicator) {
    static_assert(PassiveTargetRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy PassiveTargetRmaCommunicator");
}

TEST(ConceptSatisfactionTest, MpiRmaAdapterSatisfiesAtomicRmaCommunicator) {
    static_assert(AtomicRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy AtomicRmaCommunicator");
}

TEST(ConceptSatisfactionTest, MpiRmaAdapterSatisfiesFullRmaCommunicator) {
    static_assert(FullRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy FullRmaCommunicator");
}

TEST(ConceptSatisfactionTest, CapabilityTraitsWork) {
    EXPECT_TRUE(supports_rma_v<mpi::mpi_rma_adapter>);
    EXPECT_TRUE(supports_passive_target_rma_v<mpi::mpi_rma_adapter>);
    EXPECT_TRUE(supports_atomic_rma_v<mpi::mpi_rma_adapter>);
    EXPECT_TRUE(supports_full_rma_v<mpi::mpi_rma_adapter>);
}
#else
TEST(ConceptSatisfactionTest, CapabilityTraitsWithoutMpi) {
    // Without MPI, the adapter may not satisfy concepts
    // Just verify the code compiles
    SUCCEED();
}
#endif

// =============================================================================
// Lock Mode Tests (for MPI backend)
// =============================================================================

TEST(MpiLockModeTest, ExclusiveModeValue) {
    auto mode = rma_lock_mode::exclusive;
    EXPECT_EQ(mode, rma_lock_mode::exclusive);
}

TEST(MpiLockModeTest, SharedModeValue) {
    auto mode = rma_lock_mode::shared;
    EXPECT_EQ(mode, rma_lock_mode::shared);
}

TEST(MpiLockModeTest, ModesAreDifferent) {
    EXPECT_NE(rma_lock_mode::exclusive, rma_lock_mode::shared);
}

// =============================================================================
// Window Handle Tests (for MPI backend)
// =============================================================================

TEST(MpiWindowHandleTest, DefaultIsInvalid) {
    window_handle handle;
    EXPECT_FALSE(handle.valid());
    EXPECT_EQ(handle.handle, nullptr);
}

TEST(MpiWindowHandleTest, ValidHandleWithValue) {
    int dummy = 0;
    window_handle handle{&dummy};
    EXPECT_TRUE(handle.valid());
    EXPECT_EQ(handle.handle, &dummy);
}

TEST(MpiWindowHandleTest, EqualityComparison) {
    int dummy1 = 0;
    int dummy2 = 0;
    window_handle h1{&dummy1};
    window_handle h2{&dummy1};
    window_handle h3{&dummy2};

    EXPECT_EQ(h1, h2);
    EXPECT_NE(h1, h3);
}

}  // namespace dtl::test
