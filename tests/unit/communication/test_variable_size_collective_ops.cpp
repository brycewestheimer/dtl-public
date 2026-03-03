// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_variable_size_collective_ops.cpp
/// @brief Unit tests for variable-size collective free functions (CR-P03-T07)
/// @details Tests scatterv, gatherv, allgatherv, alltoallv with null_communicator.

#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/communicator_base.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace dtl::test {

// ============================================================================
// Test Fixture
// ============================================================================

class VCollectiveOpsTest : public ::testing::Test {
protected:
    null_communicator comm;
};

// ============================================================================
// Scatterv Tests
// ============================================================================

TEST_F(VCollectiveOpsTest, Scatterv_SingleRank_UniformData) {
    // Single rank: root scatters all data to itself
    std::vector<int> send = {10, 20, 30, 40, 50};
    std::vector<size_type> counts = {5};
    std::vector<size_type> displs = {0};
    std::vector<int> recv(5, 0);

    auto result = scatterv(comm, std::span<const int>(send),
                           std::span<const size_type>(counts),
                           std::span<const size_type>(displs),
                           std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(VCollectiveOpsTest, Scatterv_SingleElement) {
    std::vector<int> send = {42};
    std::vector<size_type> counts = {1};
    std::vector<size_type> displs = {0};
    std::vector<int> recv(1, 0);

    auto result = scatterv(comm, std::span<const int>(send),
                           std::span<const size_type>(counts),
                           std::span<const size_type>(displs),
                           std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 42);
}

TEST_F(VCollectiveOpsTest, Scatterv_DoubleData) {
    std::vector<double> send = {1.1, 2.2, 3.3};
    std::vector<size_type> counts = {3};
    std::vector<size_type> displs = {0};
    std::vector<double> recv(3, 0.0);

    auto result = scatterv(comm, std::span<const double>(send),
                           std::span<const size_type>(counts),
                           std::span<const size_type>(displs),
                           std::span<double>(recv), 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Gatherv Tests
// ============================================================================

TEST_F(VCollectiveOpsTest, Gatherv_SingleRank_AllData) {
    std::vector<int> send = {10, 20, 30, 40};
    std::vector<int> recv(4, 0);
    std::vector<size_type> counts = {4};
    std::vector<size_type> displs = {0};

    auto result = gatherv(comm, std::span<const int>(send),
                          std::span<int>(recv),
                          std::span<const size_type>(counts),
                          std::span<const size_type>(displs), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(VCollectiveOpsTest, Gatherv_SingleElement) {
    std::vector<int> send = {99};
    std::vector<int> recv(1, 0);
    std::vector<size_type> counts = {1};
    std::vector<size_type> displs = {0};

    auto result = gatherv(comm, std::span<const int>(send),
                          std::span<int>(recv),
                          std::span<const size_type>(counts),
                          std::span<const size_type>(displs), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 99);
}

TEST_F(VCollectiveOpsTest, Gatherv_DoubleData) {
    std::vector<double> send = {1.5, 2.5, 3.5};
    std::vector<double> recv(3, 0.0);
    std::vector<size_type> counts = {3};
    std::vector<size_type> displs = {0};

    auto result = gatherv(comm, std::span<const double>(send),
                          std::span<double>(recv),
                          std::span<const size_type>(counts),
                          std::span<const size_type>(displs), 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(VCollectiveOpsTest, Gatherv_LongData) {
    std::vector<long> send = {100L, 200L, 300L, 400L, 500L};
    std::vector<long> recv(5, 0L);
    std::vector<size_type> counts = {5};
    std::vector<size_type> displs = {0};

    auto result = gatherv(comm, std::span<const long>(send),
                          std::span<long>(recv),
                          std::span<const size_type>(counts),
                          std::span<const size_type>(displs), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

// ============================================================================
// Allgatherv Tests
// ============================================================================

TEST_F(VCollectiveOpsTest, Allgatherv_SingleRank) {
    std::vector<int> send = {1, 2, 3};
    std::vector<int> recv(3, 0);
    std::vector<size_type> counts = {3};
    std::vector<size_type> displs = {0};

    auto result = allgatherv(comm, std::span<const int>(send),
                             std::span<int>(recv),
                             std::span<const size_type>(counts),
                             std::span<const size_type>(displs));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(VCollectiveOpsTest, Allgatherv_DoubleData) {
    std::vector<double> send = {3.14, 2.72};
    std::vector<double> recv(2, 0.0);
    std::vector<size_type> counts = {2};
    std::vector<size_type> displs = {0};

    auto result = allgatherv(comm, std::span<const double>(send),
                             std::span<double>(recv),
                             std::span<const size_type>(counts),
                             std::span<const size_type>(displs));
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(VCollectiveOpsTest, Allgatherv_SingleElement) {
    std::vector<int> send = {7};
    std::vector<int> recv(1, 0);
    std::vector<size_type> counts = {1};
    std::vector<size_type> displs = {0};

    auto result = allgatherv(comm, std::span<const int>(send),
                             std::span<int>(recv),
                             std::span<const size_type>(counts),
                             std::span<const size_type>(displs));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 7);
}

// ============================================================================
// Alltoallv Tests
// ============================================================================

TEST_F(VCollectiveOpsTest, Alltoallv_SingleRank) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    std::vector<size_type> send_counts = {3};
    std::vector<size_type> send_displs = {0};
    std::vector<size_type> recv_counts = {3};
    std::vector<size_type> recv_displs = {0};

    auto result = alltoallv(comm, std::span<const int>(send),
                            std::span<const size_type>(send_counts),
                            std::span<const size_type>(send_displs),
                            std::span<int>(recv),
                            std::span<const size_type>(recv_counts),
                            std::span<const size_type>(recv_displs));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(VCollectiveOpsTest, Alltoallv_DoubleData) {
    std::vector<double> send = {1.0, 2.0};
    std::vector<double> recv(2, 0.0);
    std::vector<size_type> send_counts = {2};
    std::vector<size_type> send_displs = {0};
    std::vector<size_type> recv_counts = {2};
    std::vector<size_type> recv_displs = {0};

    auto result = alltoallv(comm, std::span<const double>(send),
                            std::span<const size_type>(send_counts),
                            std::span<const size_type>(send_displs),
                            std::span<double>(recv),
                            std::span<const size_type>(recv_counts),
                            std::span<const size_type>(recv_displs));
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(VCollectiveOpsTest, Gatherv_LargeData) {
    // Test with a larger dataset
    constexpr size_t N = 1000;
    std::vector<int> send(N);
    std::iota(send.begin(), send.end(), 0);
    std::vector<int> recv(N, -1);
    std::vector<size_type> counts = {N};
    std::vector<size_type> displs = {0};

    auto result = gatherv(comm, std::span<const int>(send),
                          std::span<int>(recv),
                          std::span<const size_type>(counts),
                          std::span<const size_type>(displs), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(VCollectiveOpsTest, Scatterv_LargeData) {
    constexpr size_t N = 1000;
    std::vector<int> send(N);
    std::iota(send.begin(), send.end(), 0);
    std::vector<int> recv(N, -1);
    std::vector<size_type> counts = {N};
    std::vector<size_type> displs = {0};

    auto result = scatterv(comm, std::span<const int>(send),
                           std::span<const size_type>(counts),
                           std::span<const size_type>(displs),
                           std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

}  // namespace dtl::test
