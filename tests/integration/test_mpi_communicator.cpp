// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_mpi_communicator.cpp
/// @brief Integration tests for MPI communicator
/// @details Tests MPI communication operations with multiple ranks.
/// @note Run with: mpirun -np 2 ./test_executable
///       or:       mpirun -np 4 ./test_executable

#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_communicator.hpp>
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_MPI

// =============================================================================
// Test Fixture
// =============================================================================

class MpiCommunicatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // MPI should already be initialized by the test framework
        comm_ = &mpi::world_communicator();
        adapter_ = std::make_unique<mpi::mpi_comm_adapter>(*comm_);
    }

    mpi::mpi_communicator* comm_ = nullptr;
    std::unique_ptr<mpi::mpi_comm_adapter> adapter_;
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, AdapterSatisfiesCommunicator) {
    static_assert(Communicator<mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy Communicator concept");
}

TEST_F(MpiCommunicatorTest, AdapterSatisfiesCollectiveCommunicator) {
    static_assert(CollectiveCommunicator<mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy CollectiveCommunicator concept");
}

TEST_F(MpiCommunicatorTest, AdapterSatisfiesReducingCommunicator) {
    static_assert(ReducingCommunicator<mpi::mpi_comm_adapter>,
                  "mpi_comm_adapter must satisfy ReducingCommunicator concept");
}

// =============================================================================
// Basic Properties Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, RankInValidRange) {
    EXPECT_GE(adapter_->rank(), 0);
    EXPECT_LT(adapter_->rank(), adapter_->size());
}

TEST_F(MpiCommunicatorTest, SizeAtLeastOne) {
    EXPECT_GE(adapter_->size(), 1);
}

TEST_F(MpiCommunicatorTest, CommunicatorValid) {
    EXPECT_TRUE(comm_->valid());
}

// =============================================================================
// Point-to-Point Communication Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, SendRecvPingPong) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    constexpr int tag = 42;
    int value = 0;

    if (adapter_->rank() == 0) {
        // Rank 0 sends to rank 1, then receives from rank 1
        int send_value = 123;
        adapter_->send(&send_value, sizeof(int), 1, tag);
        adapter_->recv(&value, sizeof(int), 1, tag);
        EXPECT_EQ(value, 456);
    } else if (adapter_->rank() == 1) {
        // Rank 1 receives from rank 0, then sends to rank 0
        adapter_->recv(&value, sizeof(int), 0, tag);
        EXPECT_EQ(value, 123);
        int send_value = 456;
        adapter_->send(&send_value, sizeof(int), 0, tag);
    }

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, IsendIrecvWithWait) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    constexpr int tag = 100;
    int send_value = 0;
    int recv_value = 0;

    if (adapter_->rank() == 0) {
        send_value = 789;
        auto send_req = adapter_->isend(&send_value, sizeof(int), 1, tag);
        auto recv_req = adapter_->irecv(&recv_value, sizeof(int), 1, tag + 1);

        adapter_->wait(send_req);
        adapter_->wait(recv_req);

        EXPECT_EQ(recv_value, 987);
    } else if (adapter_->rank() == 1) {
        auto recv_req = adapter_->irecv(&recv_value, sizeof(int), 0, tag);
        send_value = 987;
        auto send_req = adapter_->isend(&send_value, sizeof(int), 0, tag + 1);

        adapter_->wait(recv_req);
        adapter_->wait(send_req);

        EXPECT_EQ(recv_value, 789);
    }

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, TestCompletion) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    constexpr int tag = 200;
    int value = adapter_->rank();

    if (adapter_->rank() == 0) {
        auto req = adapter_->isend(&value, sizeof(int), 1, tag);

        // Poll until complete
        while (!adapter_->test(req)) {
            // Busy wait
        }
    } else if (adapter_->rank() == 1) {
        int recv_value = -1;
        auto req = adapter_->irecv(&recv_value, sizeof(int), 0, tag);

        // Poll until complete
        while (!adapter_->test(req)) {
            // Busy wait
        }

        EXPECT_EQ(recv_value, 0);
    }

    adapter_->barrier();
}

// =============================================================================
// Collective Communication Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, BarrierSynchronizes) {
    // All ranks call barrier - this should not hang
    adapter_->barrier();
    adapter_->barrier();
    adapter_->barrier();
    SUCCEED();
}

TEST_F(MpiCommunicatorTest, BroadcastFromRoot) {
    std::vector<int> data(10);

    if (adapter_->rank() == 0) {
        std::iota(data.begin(), data.end(), 100);  // 100, 101, ..., 109
    }

    adapter_->broadcast(data.data(), data.size() * sizeof(int), 0);

    // All ranks should have the same data
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i], static_cast<int>(100 + i));
    }
}

TEST_F(MpiCommunicatorTest, BroadcastFromNonRoot) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    int value = 0;
    const rank_t root = 1;

    if (adapter_->rank() == root) {
        value = 999;
    }

    adapter_->broadcast(&value, sizeof(int), root);

    EXPECT_EQ(value, 999);
}

TEST_F(MpiCommunicatorTest, GatherToRoot) {
    int send_value = adapter_->rank() * 10;  // Rank 0 sends 0, rank 1 sends 10, etc.
    std::vector<int> recv_buffer(static_cast<std::size_t>(adapter_->size()));

    adapter_->gather(&send_value, recv_buffer.data(), sizeof(int), 0);

    if (adapter_->rank() == 0) {
        for (rank_t r = 0; r < adapter_->size(); ++r) {
            EXPECT_EQ(recv_buffer[static_cast<std::size_t>(r)], r * 10);
        }
    }
}

TEST_F(MpiCommunicatorTest, ScatterFromRoot) {
    std::vector<int> send_buffer(static_cast<std::size_t>(adapter_->size()));
    int recv_value = -1;

    if (adapter_->rank() == 0) {
        for (rank_t r = 0; r < adapter_->size(); ++r) {
            send_buffer[static_cast<std::size_t>(r)] = r * 100;
        }
    }

    adapter_->scatter(send_buffer.data(), &recv_value, sizeof(int), 0);

    EXPECT_EQ(recv_value, adapter_->rank() * 100);
}

TEST_F(MpiCommunicatorTest, Allgather) {
    int send_value = adapter_->rank() + 1;  // Rank 0 sends 1, rank 1 sends 2, etc.
    std::vector<int> recv_buffer(static_cast<std::size_t>(adapter_->size()));

    adapter_->allgather(&send_value, recv_buffer.data(), sizeof(int));

    for (rank_t r = 0; r < adapter_->size(); ++r) {
        EXPECT_EQ(recv_buffer[static_cast<std::size_t>(r)], r + 1);
    }
}

TEST_F(MpiCommunicatorTest, Alltoall) {
    const size_type count = static_cast<size_type>(adapter_->size());
    std::vector<int> send_buffer(count);
    std::vector<int> recv_buffer(count);

    // Each rank sends its rank number to all ranks
    for (size_type i = 0; i < count; ++i) {
        send_buffer[i] = adapter_->rank() * 1000 + static_cast<int>(i);
    }

    adapter_->alltoall(send_buffer.data(), recv_buffer.data(), sizeof(int));

    // Each position i contains what rank i sent to this rank
    for (rank_t r = 0; r < adapter_->size(); ++r) {
        EXPECT_EQ(recv_buffer[static_cast<std::size_t>(r)], r * 1000 + adapter_->rank());
    }
}

// =============================================================================
// Reduction Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, ReduceSum) {
    double local_value = static_cast<double>(adapter_->rank() + 1);  // 1, 2, 3, ...
    double result = 0.0;

    adapter_->reduce_sum(&local_value, &result, 1, 0);

    if (adapter_->rank() == 0) {
        // Sum of 1..n = n*(n+1)/2
        double expected = adapter_->size() * (adapter_->size() + 1) / 2.0;
        EXPECT_DOUBLE_EQ(result, expected);
    }
}

TEST_F(MpiCommunicatorTest, AllreduceSum) {
    double local_value = static_cast<double>(adapter_->rank() + 1);
    double result = 0.0;

    adapter_->allreduce_sum(&local_value, &result, 1);

    // All ranks should have the same result
    double expected = adapter_->size() * (adapter_->size() + 1) / 2.0;
    EXPECT_DOUBLE_EQ(result, expected);
}

TEST_F(MpiCommunicatorTest, ReduceSumArray) {
    constexpr size_type count = 5;
    std::vector<double> local_data(count);
    std::vector<double> result(count, 0.0);

    for (size_type i = 0; i < count; ++i) {
        local_data[i] = static_cast<double>(adapter_->rank() + 1) * static_cast<double>(i + 1);
    }

    adapter_->reduce_sum(local_data.data(), result.data(), count, 0);

    if (adapter_->rank() == 0) {
        // Sum over ranks: for element i, sum is (i+1) * sum(1..n) = (i+1) * n*(n+1)/2
        double rank_sum = adapter_->size() * (adapter_->size() + 1) / 2.0;
        for (size_type i = 0; i < count; ++i) {
            EXPECT_DOUBLE_EQ(result[i], static_cast<double>(i + 1) * rank_sum);
        }
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, InvalidRequestTest) {
    request_handle invalid_req;  // handle is nullptr by default

    // Test on invalid request should return true (considered complete)
    EXPECT_TRUE(adapter_->test(invalid_req));
}

// =============================================================================
// Multi-Rank Coordination Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, RingCommunication) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    constexpr int tag = 300;
    int send_value = adapter_->rank();
    int recv_value = -1;

    rank_t next_rank = (adapter_->rank() + 1) % adapter_->size();
    rank_t prev_rank = (adapter_->rank() - 1 + adapter_->size()) % adapter_->size();

    // Send to next, receive from previous (ring pattern)
    auto send_req = adapter_->isend(&send_value, sizeof(int), next_rank, tag);
    auto recv_req = adapter_->irecv(&recv_value, sizeof(int), prev_rank, tag);

    adapter_->wait(send_req);
    adapter_->wait(recv_req);

    EXPECT_EQ(recv_value, prev_rank);
}

TEST_F(MpiCommunicatorTest, LargeDataTransfer) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    constexpr size_type data_size = 1000000;  // 1M elements
    constexpr int tag = 400;

    std::vector<int> data(data_size);

    if (adapter_->rank() == 0) {
        std::iota(data.begin(), data.end(), 0);
        adapter_->send(data.data(), data.size() * sizeof(int), 1, tag);
    } else if (adapter_->rank() == 1) {
        adapter_->recv(data.data(), data.size() * sizeof(int), 0, tag);

        // Verify data
        for (size_type i = 0; i < data_size; ++i) {
            EXPECT_EQ(data[i], static_cast<int>(i)) << "Mismatch at index " << i;
            if (data[i] != static_cast<int>(i)) {
                break;  // Stop on first failure to avoid massive output
            }
        }
    }

    adapter_->barrier();
}

// =============================================================================
// Communicator Split Tests
// =============================================================================

TEST_F(MpiCommunicatorTest, SplitByColorEvenOdd) {
    if (adapter_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 ranks";
    }

    // Split into even/odd ranks
    int color = adapter_->rank() % 2;  // 0 for even, 1 for odd
    int key = adapter_->rank();         // Preserve ordering within color

    auto split_adapter = adapter_->split(color, key);

    // Verify split communicator properties
    EXPECT_GE(split_adapter.rank(), 0);
    EXPECT_LT(split_adapter.rank(), split_adapter.size());
    EXPECT_GT(split_adapter.size(), 0);

    // Expected size: half of world size (rounded up for odd)
    int expected_size = (adapter_->size() + color) / 2;
    EXPECT_EQ(split_adapter.size(), expected_size);

    // Verify ranks within split communicator are correctly ordered
    std::vector<rank_t> all_ranks(static_cast<std::size_t>(split_adapter.size()));
    rank_t my_world_rank = adapter_->rank();
    split_adapter.allgather(&my_world_rank, all_ranks.data(), sizeof(rank_t));

    // All ranks in the split comm should have same color (even/odd)
    for (rank_t r : all_ranks) {
        EXPECT_EQ(r % 2, color) << "Rank " << r << " has wrong color";
    }

    // Verify collective operation in split communicator
    int local_value = 1;
    int sum = split_adapter.allreduce_sum_value(local_value);
    EXPECT_EQ(sum, split_adapter.size());

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, SplitWithCustomKey) {
    if (adapter_->size() < 4) {
        GTEST_SKIP() << "Requires at least 4 ranks";
    }

    // Split into two groups: ranks 0,1 and ranks 2,3+
    int color = (adapter_->rank() < 2) ? 0 : 1;
    // Reverse ordering within each group
    int key = -adapter_->rank();

    auto split_adapter = adapter_->split(color, key);

    // Verify ranks are correctly ordered (reversed) within group
    std::vector<rank_t> all_ranks(static_cast<std::size_t>(split_adapter.size()));
    rank_t my_world_rank = adapter_->rank();
    split_adapter.allgather(&my_world_rank, all_ranks.data(), sizeof(rank_t));

    // Within each group, ranks should be in descending order
    for (size_t i = 1; i < all_ranks.size(); ++i) {
        EXPECT_GT(all_ranks[i-1], all_ranks[i])
            << "Ranks not properly ordered by key";
    }

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, SplitSingletonCommunicator) {
    // Each rank in its own communicator
    int color = adapter_->rank();
    int key = 0;

    auto split_adapter = adapter_->split(color, key);

    // Should be a singleton communicator
    EXPECT_EQ(split_adapter.size(), 1);
    EXPECT_EQ(split_adapter.rank(), 0);

    // Verify operations still work in singleton
    int value = 42;
    int result = split_adapter.allreduce_sum_value(value);
    EXPECT_EQ(result, 42);

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, SplitCommunicatorOwnership) {
    // Test that split communicators manage ownership correctly
    int color = adapter_->rank() % 2;

    {
        // Create split adapter in inner scope
        auto split1 = adapter_->split(color, 0);
        EXPECT_EQ(split1.rank(), adapter_->rank() / 2);

        // Copy split adapter (shared ownership)
        auto split2 = split1;
        EXPECT_EQ(split1.rank(), split2.rank());
        EXPECT_EQ(split1.size(), split2.size());

        // Both should still be usable
        EXPECT_NO_THROW(split1.barrier());
        EXPECT_NO_THROW(split2.barrier());
    }
    // Split communicator should be cleaned up here

    adapter_->barrier();
}

TEST_F(MpiCommunicatorTest, SplitMultipleGroups) {
    if (adapter_->size() < 4) {
        GTEST_SKIP() << "Requires at least 4 ranks";
    }

    // Split into 4 groups based on rank mod 4
    int color = adapter_->rank() % 4;
    int key = adapter_->rank();

    auto split_adapter = adapter_->split(color, key);

    // Each group should have roughly 1/4 of the ranks
    EXPECT_GT(split_adapter.size(), 0);
    EXPECT_LE(split_adapter.size(), (adapter_->size() + 3) / 4);

    // Verify all ranks in the group have the same color
    std::vector<rank_t> all_ranks(static_cast<std::size_t>(split_adapter.size()));
    rank_t my_world_rank = adapter_->rank();
    split_adapter.allgather(&my_world_rank, all_ranks.data(), sizeof(rank_t));

    for (rank_t r : all_ranks) {
        EXPECT_EQ(r % 4, color) << "Rank " << r << " has wrong color";
    }

    adapter_->barrier();
}

#else  // !DTL_ENABLE_MPI

TEST(MpiCommunicatorTest, MpiNotEnabled) {
    GTEST_SKIP() << "MPI not enabled - skipping MPI tests";
}

#endif  // DTL_ENABLE_MPI

}  // namespace dtl::test

// =============================================================================
// Main (with MPI initialization)
// =============================================================================

#if DTL_ENABLE_MPI
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
#else
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
