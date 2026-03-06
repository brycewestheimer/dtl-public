// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_rma_mpi.cpp
/// @brief Integration tests for MPI RMA operations
/// @details Tests RMA operations with multiple ranks.
/// @note Run with: mpirun -np 2 ./dtl_mpi_tests --gtest_filter="RmaMpi*"
///       or:       mpirun -np 4 ./dtl_mpi_tests --gtest_filter="RmaMpi*"

#include <dtl/backend/concepts/rma_communicator.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/mpi/mpi_window.hpp>
#include <backends/mpi/mpi_rma_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>
#include <array>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture
// =============================================================================

class RmaMpiTest : public ::testing::Test {
protected:
    void SetUp() override {
        mpi_domain_ = std::make_unique<mpi_domain>();
        auto& comm = mpi_domain_->communicator().underlying();
        adapter_ = std::make_unique<mpi::mpi_rma_adapter>(comm);
        rank_ = mpi_domain_->rank();
        size_ = mpi_domain_->size();
    }

    void TearDown() override {
        // Clean up any remaining windows
        for (auto& win : windows_) {
            if (win.valid()) {
                adapter_->free_window(win);
            }
        }
        windows_.clear();
        adapter_.reset();
        mpi_domain_.reset();
    }

    // Helper to create and track a window
    window_handle create_tracked_window(void* base, size_type size) {
        auto win = adapter_->create_window(base, size);
        windows_.push_back(win);
        return win;
    }

    std::unique_ptr<mpi_domain> mpi_domain_;
    std::unique_ptr<mpi::mpi_rma_adapter> adapter_;
    rank_t rank_ = 0;
    rank_t size_ = 0;
    std::vector<window_handle> windows_;
};

// =============================================================================
// Concept Verification
// =============================================================================

TEST_F(RmaMpiTest, AdapterSatisfiesRmaConcepts) {
    static_assert(RmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy RmaCommunicator");
    static_assert(PassiveTargetRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy PassiveTargetRmaCommunicator");
    static_assert(AtomicRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy AtomicRmaCommunicator");
    static_assert(FullRmaCommunicator<mpi::mpi_rma_adapter>,
                  "mpi_rma_adapter must satisfy FullRmaCommunicator");
}

// =============================================================================
// Window Creation Tests
// =============================================================================

TEST_F(RmaMpiTest, WindowCreationCollective) {
    // All ranks must participate in window creation
    std::array<int, 10> data{};
    data.fill(rank_);

    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));
    EXPECT_TRUE(win.valid());

    adapter_->fence(win);  // Synchronize
    adapter_->fence(win);  // Complete
}

TEST_F(RmaMpiTest, WindowCreationMultiple) {
    std::array<int, 10> data1{};
    std::array<int, 10> data2{};

    auto win1 = create_tracked_window(data1.data(), data1.size() * sizeof(int));
    auto win2 = create_tracked_window(data2.data(), data2.size() * sizeof(int));

    EXPECT_TRUE(win1.valid());
    EXPECT_TRUE(win2.valid());
    EXPECT_NE(win1.handle, win2.handle);

    adapter_->fence(win1);
    adapter_->fence(win2);
    adapter_->fence(win1);
    adapter_->fence(win2);
}

// =============================================================================
// Put/Get with Fence Synchronization
// =============================================================================

TEST_F(RmaMpiTest, PutGetWithFence) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    // Open epoch
    adapter_->fence(win);

    // Rank 0 puts to rank 1
    if (rank_ == 0) {
        std::array<int, 5> send_data = {10, 20, 30, 40, 50};
        adapter_->put(send_data.data(), send_data.size() * sizeof(int), 1, 0, win);
    }

    // Close epoch - all operations complete
    adapter_->fence(win);

    // Rank 1 should have received the data
    if (rank_ == 1) {
        EXPECT_EQ(data[0], 10);
        EXPECT_EQ(data[1], 20);
        EXPECT_EQ(data[2], 30);
        EXPECT_EQ(data[3], 40);
        EXPECT_EQ(data[4], 50);
    }
}

TEST_F(RmaMpiTest, GetWithFence) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    if (rank_ == 0) {
        data = {100, 200, 300, 400, 500, 0, 0, 0, 0, 0};
    }

    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    adapter_->fence(win);

    // Rank 1 gets from rank 0
    std::array<int, 5> recv_data{};
    if (rank_ == 1) {
        adapter_->get(recv_data.data(), recv_data.size() * sizeof(int), 0, 0, win);
    }

    adapter_->fence(win);

    if (rank_ == 1) {
        EXPECT_EQ(recv_data[0], 100);
        EXPECT_EQ(recv_data[1], 200);
        EXPECT_EQ(recv_data[2], 300);
        EXPECT_EQ(recv_data[3], 400);
        EXPECT_EQ(recv_data[4], 500);
    }
}

TEST_F(RmaMpiTest, RingPut) {
    // Each rank puts to the next rank in a ring
    std::array<int, 1> data{};
    data[0] = -1;  // Initial value

    auto win = create_tracked_window(data.data(), sizeof(int));

    adapter_->fence(win);

    // Put rank to next rank (modulo size)
    int send_value = rank_;
    rank_t target = (rank_ + 1) % size_;
    adapter_->put(&send_value, sizeof(int), target, 0, win);

    adapter_->fence(win);

    // Each rank should have received the previous rank's value
    int expected = (rank_ == 0) ? (size_ - 1) : (rank_ - 1);
    EXPECT_EQ(data[0], expected);
}

// =============================================================================
// Lock/Unlock Passive-Target Tests
// =============================================================================

TEST_F(RmaMpiTest, LockUnlockSingleTarget) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    // Passive-target epochs should not be mixed with fence epochs.
    adapter_->comm().barrier();

    // Rank 0 does a lock/unlock access to rank 1
    if (rank_ == 0) {
        adapter_->lock(1, rma_lock_mode::exclusive, win);

        int value = 999;
        adapter_->put(&value, sizeof(int), 1, 0, win);
        adapter_->flush(1, win);

        adapter_->unlock(1, win);
    }

    // Barrier to ensure rank 0's operations complete before rank 1 checks
    adapter_->comm().barrier();

    if (rank_ == 1) {
        EXPECT_EQ(data[0], 999);
    }
}

TEST_F(RmaMpiTest, LockAllSharedAccess) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    data.fill(rank_ * 10);

    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    // Passive-target epochs should not be mixed with fence epochs.
    adapter_->comm().barrier();

    // Smoke-test that all ranks can enter and leave a shared lock_all epoch.
    adapter_->lock_all(win);
    adapter_->comm().barrier();
    adapter_->unlock_all(win);
    adapter_->comm().barrier();

    // Validate shared passive-target reads through a single-target lock.
    if (rank_ == 0) {
        int value = -1;
        adapter_->lock(1, rma_lock_mode::shared, win);
        adapter_->get(&value, sizeof(int), 1, 0, win);
        adapter_->unlock(1, win);
        EXPECT_EQ(value, 10);
    }

    adapter_->comm().barrier();
}

// =============================================================================
// Atomic Operations Tests
// =============================================================================

TEST_F(RmaMpiTest, AccumulateSum) {
    // All ranks accumulate to rank 0
    std::array<int, 1> data{};
    data[0] = 0;

    auto win = create_tracked_window(data.data(), sizeof(int));

    adapter_->fence(win);

    // Each rank adds its rank value to rank 0
    if (rank_ != 0) {
        int value = rank_;
        adapter_->accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::sum, win);
    }

    adapter_->fence(win);

    // Rank 0 should have sum of all ranks: 0 + 1 + 2 + ... + (size-1)
    if (rank_ == 0) {
        int expected_sum = (size_ * (size_ - 1)) / 2;
        EXPECT_EQ(data[0], expected_sum);
    }
}

TEST_F(RmaMpiTest, AccumulateMax) {
    std::array<int, 1> data{};
    data[0] = rank_;

    auto win = create_tracked_window(data.data(), sizeof(int));

    adapter_->fence(win);

    // All ranks put their rank to rank 0 using MAX
    int value = rank_;
    adapter_->accumulate(&value, sizeof(int), 0, 0, rma_reduce_op::max, win);

    adapter_->fence(win);

    // Rank 0 should have the maximum rank value
    if (rank_ == 0) {
        EXPECT_EQ(data[0], size_ - 1);
    }
}

// =============================================================================
// Large Transfer Tests
// =============================================================================

TEST_F(RmaMpiTest, LargeTransfer) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    // 1 MB transfer
    constexpr size_t num_elements = 1024 * 256;  // 256K ints = 1MB
    std::vector<int> data(num_elements, 0);

    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    adapter_->fence(win);

    // Rank 0 sends 1MB to rank 1
    if (rank_ == 0) {
        std::vector<int> send_data(num_elements);
        std::iota(send_data.begin(), send_data.end(), 0);  // 0, 1, 2, ...
        adapter_->put(send_data.data(), send_data.size() * sizeof(int), 1, 0, win);
    }

    adapter_->fence(win);

    if (rank_ == 1) {
        // Verify first and last elements
        EXPECT_EQ(data[0], 0);
        EXPECT_EQ(data[num_elements - 1], static_cast<int>(num_elements - 1));

        // Verify a few random elements
        EXPECT_EQ(data[1000], 1000);
        EXPECT_EQ(data[100000], 100000);
    }
}

// =============================================================================
// Flush Guarantee Tests
// =============================================================================

TEST_F(RmaMpiTest, FlushGuaranteesCompletion) {
    if (size_ < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    std::array<int, 10> data{};
    auto win = create_tracked_window(data.data(), data.size() * sizeof(int));

    adapter_->fence(win);

    // Rank 0 locks, puts, flushes, and verifies rank 1 can see the data
    if (rank_ == 0) {
        adapter_->lock(1, rma_lock_mode::exclusive, win);

        // Put value
        int value = 12345;
        adapter_->put(&value, sizeof(int), 1, 0, win);

        // Flush guarantees remote completion
        adapter_->flush(1, win);

        adapter_->unlock(1, win);
    }

    adapter_->comm().barrier();

    if (rank_ == 1) {
        EXPECT_EQ(data[0], 12345);
    }
}

// =============================================================================
// Multiple Windows Test
// =============================================================================

TEST_F(RmaMpiTest, MultipleWindowsIndependent) {
    std::array<int, 10> data1{};
    std::array<int, 10> data2{};
    data1.fill(0);
    data2.fill(0);

    auto win1 = create_tracked_window(data1.data(), data1.size() * sizeof(int));
    auto win2 = create_tracked_window(data2.data(), data2.size() * sizeof(int));

    adapter_->fence(win1);
    adapter_->fence(win2);

    // Put different values to different windows
    if (rank_ == 0 && size_ >= 2) {
        int val1 = 111;
        int val2 = 222;
        adapter_->put(&val1, sizeof(int), 1, 0, win1);
        adapter_->put(&val2, sizeof(int), 1, 0, win2);
    }

    adapter_->fence(win1);
    adapter_->fence(win2);

    if (rank_ == 1) {
        EXPECT_EQ(data1[0], 111);
        EXPECT_EQ(data2[0], 222);
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(RmaMpiTest, WindowFreeIsSafe) {
    std::array<int, 10> data{};
    auto win = adapter_->create_window(data.data(), data.size() * sizeof(int));

    adapter_->fence(win);
    adapter_->fence(win);

    // Free should not throw
    EXPECT_NO_THROW(adapter_->free_window(win));

    // Free again should be safe (handle is now null)
    EXPECT_NO_THROW(adapter_->free_window(win));
}

#else  // !DTL_ENABLE_MPI

// When MPI is not enabled, provide a placeholder test
TEST(RmaMpiTest, MpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled";
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test
