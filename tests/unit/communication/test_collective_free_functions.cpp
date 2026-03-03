// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_collective_free_functions.cpp
/// @brief Comprehensive unit tests for collective free functions (CR-P03-T06)
/// @details Tests all collective operations with null_communicator (single-rank).

#include <dtl/communication/collective_ops.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/communication/reduction_ops.hpp>
#include <dtl/communication/point_to_point.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <numeric>
#include <vector>

namespace dtl::comm_test {

// ============================================================================
// Test Fixture
// ============================================================================

class CollectiveFreeTest : public ::testing::Test {
protected:
    null_communicator comm;
};

// ============================================================================
// Barrier Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Barrier_Succeeds) {
    auto result = barrier(comm);
    ASSERT_TRUE(result.has_value());
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Broadcast_SpanInt) {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto result = broadcast(comm, std::span<int>(data), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(data, (std::vector<int>{1, 2, 3, 4, 5}));
}

TEST_F(CollectiveFreeTest, Broadcast_SingleValue) {
    double value = 3.14;
    auto result = broadcast(comm, value, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_DOUBLE_EQ(value, 3.14);
}

TEST_F(CollectiveFreeTest, Broadcast_SpanDouble) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    auto result = broadcast(comm, std::span<double>(data), 0);
    ASSERT_TRUE(result.has_value());
}

// ============================================================================
// Scatter Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Scatter_IntData) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    auto result = scatter(comm, std::span<const int>(send),
                          std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scatter_DoubleData) {
    std::vector<double> send = {1.1, 2.2};
    std::vector<double> recv(2, 0.0);
    auto result = scatter(comm, std::span<const double>(send),
                          std::span<double>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scatter_SingleElement) {
    std::vector<int> send = {42};
    std::vector<int> recv(1, 0);
    auto result = scatter(comm, std::span<const int>(send),
                          std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 42);
}

// ============================================================================
// Gather Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Gather_IntData) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    auto result = gather(comm, std::span<const int>(send),
                         std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Gather_DoubleData) {
    std::vector<double> send = {1.5, 2.5, 3.5};
    std::vector<double> recv(3, 0.0);
    auto result = gather(comm, std::span<const double>(send),
                         std::span<double>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Gather_SingleElement) {
    std::vector<long> send = {99L};
    std::vector<long> recv(1, 0L);
    auto result = gather(comm, std::span<const long>(send),
                         std::span<long>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 99L);
}

// ============================================================================
// Allgather Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Allgather_IntData) {
    std::vector<int> send = {1, 2, 3};
    std::vector<int> recv(3, 0);
    auto result = allgather(comm, std::span<const int>(send),
                            std::span<int>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allgather_DoubleData) {
    std::vector<double> send = {4.0, 5.0};
    std::vector<double> recv(2, 0.0);
    auto result = allgather(comm, std::span<const double>(send),
                            std::span<double>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allgather_LongData) {
    std::vector<long> send = {100L, 200L, 300L, 400L};
    std::vector<long> recv(4, 0L);
    auto result = allgather(comm, std::span<const long>(send),
                            std::span<long>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

// ============================================================================
// Alltoall Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Alltoall_IntData) {
    std::vector<int> send = {1, 2, 3};
    std::vector<int> recv(3, 0);
    auto result = alltoall(comm, std::span<const int>(send),
                           std::span<int>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

// ============================================================================
// Reduce Tests — Sum
// ============================================================================

TEST_F(CollectiveFreeTest, Reduce_Sum_Int) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_sum<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Reduce_Sum_Double) {
    std::vector<double> send = {1.5, 2.5, 3.5};
    std::vector<double> recv(3, 0.0);
    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_sum<double>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Sum_SingleValue) {
    auto result = reduce(comm, 42, reduce_sum<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

// ============================================================================
// Reduce Tests — Min
// ============================================================================

TEST_F(CollectiveFreeTest, Reduce_Min_Int) {
    std::vector<int> send = {5, 3, 8, 1, 9};
    std::vector<int> recv(5, -1);
    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_min<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Min_Double) {
    std::vector<double> send = {1.5, 0.5, 3.5};
    std::vector<double> recv(3, -1.0);
    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_min<double>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Min_SingleValue) {
    auto result = reduce(comm, 7, reduce_min<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 7);
}

// ============================================================================
// Reduce Tests — Max
// ============================================================================

TEST_F(CollectiveFreeTest, Reduce_Max_Int) {
    std::vector<int> send = {5, 3, 8, 1, 9};
    std::vector<int> recv(5, -1);
    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_max<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Max_Double) {
    std::vector<double> send = {1.5, 2.5, 3.5};
    std::vector<double> recv(3, -1.0);
    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_max<double>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Reduce Tests — Product
// ============================================================================

TEST_F(CollectiveFreeTest, Reduce_Product_Int) {
    std::vector<int> send = {2, 3, 5};
    std::vector<int> recv(3, 0);
    auto result = reduce(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_product<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    // Single rank: identity operation
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Product_Double) {
    std::vector<double> send = {2.0, 3.0, 4.0};
    std::vector<double> recv(3, 0.0);
    auto result = reduce(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_product<double>{}, 0);
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Reduce_Product_SingleValue) {
    auto result = reduce(comm, 5, reduce_product<int>{}, 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 5);
}

// ============================================================================
// Allreduce Tests — Sum
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Sum_Int) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Sum_Double) {
    std::vector<double> send = {1.5, 2.5};
    std::vector<double> recv(2, 0.0);
    auto result = allreduce(comm, std::span<const double>(send),
                            std::span<double>(recv), reduce_sum<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

TEST_F(CollectiveFreeTest, Allreduce_Sum_Long) {
    std::vector<long> send = {100L, 200L};
    std::vector<long> recv(2, 0L);
    auto result = allreduce(comm, std::span<const long>(send),
                            std::span<long>(recv), reduce_sum<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Sum_SingleValue) {
    auto result = allreduce(comm, 42, reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 42);
}

// ============================================================================
// Allreduce Tests — Min
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Min_Int) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_min<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Min_Double) {
    std::vector<double> send = {1.5, 0.5};
    std::vector<double> recv(2, 0.0);
    auto result = allreduce(comm, std::span<const double>(send),
                            std::span<double>(recv), reduce_min<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Allreduce Tests — Max
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Max_Int) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_max<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Max_Double) {
    std::vector<double> send = {1.5, 2.5};
    std::vector<double> recv(2, 0.0);
    auto result = allreduce(comm, std::span<const double>(send),
                            std::span<double>(recv), reduce_max<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Allreduce Tests — Product
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Product_Int) {
    std::vector<int> send = {2, 3, 5};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_product<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Product_Double) {
    std::vector<double> send = {2.0, 3.0};
    std::vector<double> recv(2, 0.0);
    auto result = allreduce(comm, std::span<const double>(send),
                            std::span<double>(recv), reduce_product<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Allreduce Tests — Bitwise Operations
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Band_Int) {
    std::vector<int> send = {0xFF, 0x0F, 0xAA};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_band<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: identity
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Bor_Int) {
    std::vector<int> send = {0x01, 0x02, 0x04};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_bor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Bxor_Int) {
    std::vector<int> send = {0xAA, 0x55, 0xFF};
    std::vector<int> recv(3, 0);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_bxor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Band_Long) {
    std::vector<long> send = {0xFFL, 0x0FL};
    std::vector<long> recv(2, 0L);
    auto result = allreduce(comm, std::span<const long>(send),
                            std::span<long>(recv), reduce_band<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Bor_Long) {
    std::vector<long> send = {0x01L, 0x02L};
    std::vector<long> recv(2, 0L);
    auto result = allreduce(comm, std::span<const long>(send),
                            std::span<long>(recv), reduce_bor<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allreduce_Bxor_Long) {
    std::vector<long> send = {0xAAL, 0x55L};
    std::vector<long> recv(2, 0L);
    auto result = allreduce(comm, std::span<const long>(send),
                            std::span<long>(recv), reduce_bxor<long>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

// ============================================================================
// Allreduce Tests — Logical Operations
// ============================================================================

TEST_F(CollectiveFreeTest, Allreduce_Land_Int) {
    std::vector<int> send = {1, 0, 1};
    std::vector<int> recv(3, -1);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_land<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 1);
    EXPECT_EQ(recv[1], 0);
    EXPECT_EQ(recv[2], 1);
}

TEST_F(CollectiveFreeTest, Allreduce_Lor_Int) {
    std::vector<int> send = {0, 0, 1};
    std::vector<int> recv(3, -1);
    auto result = allreduce(comm, std::span<const int>(send),
                            std::span<int>(recv), reduce_lor<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv[0], 0);
    EXPECT_EQ(recv[1], 0);
    EXPECT_EQ(recv[2], 1);
}

// ============================================================================
// Scan Tests — Sum
// ============================================================================

TEST_F(CollectiveFreeTest, Scan_Sum_Int) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, 0);
    auto result = scan(comm, std::span<const int>(send),
                       std::span<int>(recv), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: inclusive scan = identity
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scan_Sum_Double) {
    std::vector<double> send = {1.0, 2.0, 3.0};
    std::vector<double> recv(3, 0.0);
    auto result = scan(comm, std::span<const double>(send),
                       std::span<double>(recv), reduce_sum<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Scan Tests — Min
// ============================================================================

TEST_F(CollectiveFreeTest, Scan_Min_Int) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = scan(comm, std::span<const int>(send),
                       std::span<int>(recv), reduce_min<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scan_Min_Double) {
    std::vector<double> send = {1.5, 0.5, 2.5};
    std::vector<double> recv(3, 0.0);
    auto result = scan(comm, std::span<const double>(send),
                       std::span<double>(recv), reduce_min<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Scan Tests — Max
// ============================================================================

TEST_F(CollectiveFreeTest, Scan_Max_Int) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = scan(comm, std::span<const int>(send),
                       std::span<int>(recv), reduce_max<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scan_Max_Double) {
    std::vector<double> send = {1.0, 2.0, 3.0};
    std::vector<double> recv(3, 0.0);
    auto result = scan(comm, std::span<const double>(send),
                       std::span<double>(recv), reduce_max<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Scan Tests — Product
// ============================================================================

TEST_F(CollectiveFreeTest, Scan_Product_Int) {
    std::vector<int> send = {2, 3, 5};
    std::vector<int> recv(3, 0);
    auto result = scan(comm, std::span<const int>(send),
                       std::span<int>(recv), reduce_product<int>{});
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Scan_Product_Double) {
    std::vector<double> send = {2.0, 3.0, 4.0};
    std::vector<double> recv(3, 0.0);
    auto result = scan(comm, std::span<const double>(send),
                       std::span<double>(recv), reduce_product<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < send.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], send[i]);
    }
}

// ============================================================================
// Exscan Tests — Sum
// ============================================================================

TEST_F(CollectiveFreeTest, Exscan_Sum_Int) {
    std::vector<int> send = {10, 20, 30};
    std::vector<int> recv(3, -1);
    auto result = exscan(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_sum<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank (rank 0): exscan returns identity element (0 for sum)
    for (size_t i = 0; i < recv.size(); ++i) {
        EXPECT_EQ(recv[i], 0);
    }
}

TEST_F(CollectiveFreeTest, Exscan_Sum_Double) {
    std::vector<double> send = {1.0, 2.0};
    std::vector<double> recv(2, -1.0);
    auto result = exscan(comm, std::span<const double>(send),
                         std::span<double>(recv), reduce_sum<double>{});
    ASSERT_TRUE(result.has_value());
    for (size_t i = 0; i < recv.size(); ++i) {
        EXPECT_DOUBLE_EQ(recv[i], 0.0);
    }
}

// ============================================================================
// Exscan Tests — Min
// ============================================================================

TEST_F(CollectiveFreeTest, Exscan_Min_Int) {
    std::vector<int> send = {5, 3};
    std::vector<int> recv(2, 0);
    auto result = exscan(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_min<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: exscan returns identity element (max value for min)
    for (size_t i = 0; i < recv.size(); ++i) {
        EXPECT_EQ(recv[i], std::numeric_limits<int>::max());
    }
}

// ============================================================================
// Exscan Tests — Max
// ============================================================================

TEST_F(CollectiveFreeTest, Exscan_Max_Int) {
    std::vector<int> send = {5, 3};
    std::vector<int> recv(2, 0);
    auto result = exscan(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_max<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: exscan returns identity element (lowest for max)
    for (size_t i = 0; i < recv.size(); ++i) {
        EXPECT_EQ(recv[i], std::numeric_limits<int>::lowest());
    }
}

// ============================================================================
// Exscan Tests — Product
// ============================================================================

TEST_F(CollectiveFreeTest, Exscan_Product_Int) {
    std::vector<int> send = {2, 3};
    std::vector<int> recv(2, 0);
    auto result = exscan(comm, std::span<const int>(send),
                         std::span<int>(recv), reduce_product<int>{});
    ASSERT_TRUE(result.has_value());
    // Single rank: exscan returns identity element (1 for product)
    for (size_t i = 0; i < recv.size(); ++i) {
        EXPECT_EQ(recv[i], 1);
    }
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

TEST_F(CollectiveFreeTest, Sum_Convenience) {
    std::vector<int> send = {1, 2, 3};
    std::vector<int> recv(3, 0);
    auto result = sum(comm, std::span<const int>(send),
                      std::span<int>(recv), 0);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allsum_Convenience) {
    std::vector<int> send = {1, 2, 3};
    std::vector<int> recv(3, 0);
    auto result = allsum(comm, std::span<const int>(send),
                         std::span<int>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allmax_Convenience) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = allmax(comm, std::span<const int>(send),
                         std::span<int>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

TEST_F(CollectiveFreeTest, Allmin_Convenience) {
    std::vector<int> send = {5, 3, 8};
    std::vector<int> recv(3, 0);
    auto result = allmin(comm, std::span<const int>(send),
                         std::span<int>(recv));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(recv, send);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(CollectiveFreeTest, Reduce_SingleElement_AllOps) {
    // Sum
    auto r1 = reduce(comm, 42, reduce_sum<int>{}, 0);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, 42);

    // Min
    auto r2 = reduce(comm, 42, reduce_min<int>{}, 0);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, 42);

    // Max
    auto r3 = reduce(comm, 42, reduce_max<int>{}, 0);
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(*r3, 42);

    // Product
    auto r4 = reduce(comm, 42, reduce_product<int>{}, 0);
    ASSERT_TRUE(r4.has_value());
    EXPECT_EQ(*r4, 42);
}

TEST_F(CollectiveFreeTest, Allreduce_SingleValue_AllOps) {
    // Sum
    auto r1 = allreduce(comm, 42, reduce_sum<int>{});
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(*r1, 42);

    // Min
    auto r2 = allreduce(comm, 42, reduce_min<int>{});
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(*r2, 42);

    // Max
    auto r3 = allreduce(comm, 42, reduce_max<int>{});
    ASSERT_TRUE(r3.has_value());
    EXPECT_EQ(*r3, 42);

    // Product
    auto r4 = allreduce(comm, 42, reduce_product<int>{});
    ASSERT_TRUE(r4.has_value());
    EXPECT_EQ(*r4, 42);
}

// ============================================================================
// Recv Status Reporting Tests (CR-P03-T04)
// ============================================================================

TEST_F(CollectiveFreeTest, Recv_AnySource_ResolvesToZero) {
    std::vector<int> data(3, 0);
    auto result = recv(comm, std::span<int>(data), any_source, 100);
    ASSERT_TRUE(result.has_value());
    // any_source should resolve to 0 (only rank in null communicator)
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 100);
    EXPECT_EQ(result->count, 3u);
    EXPECT_EQ(result->error, 0);
}

TEST_F(CollectiveFreeTest, Recv_AnyTag_ResolvesToZero) {
    std::vector<int> data(2, 0);
    auto result = recv(comm, std::span<int>(data), 0, any_tag);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    // any_tag should resolve to 0
    EXPECT_EQ(result->tag, 0);
    EXPECT_EQ(result->count, 2u);
    EXPECT_EQ(result->error, 0);
}

TEST_F(CollectiveFreeTest, Recv_BothWildcards_Resolved) {
    std::vector<int> data(4, 0);
    auto result = recv(comm, std::span<int>(data), any_source, any_tag);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 0);
    EXPECT_EQ(result->count, 4u);
    EXPECT_EQ(result->error, 0);
}

TEST_F(CollectiveFreeTest, Recv_SpecificSourceTag_Preserved) {
    std::vector<int> data(1, 0);
    auto result = recv(comm, std::span<int>(data), 0, 42);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 42);
    EXPECT_EQ(result->count, 1u);
    EXPECT_EQ(result->error, 0);
}

TEST_F(CollectiveFreeTest, Recv_SingleValue_StatusCorrect) {
    int value = 0;
    auto result = recv(comm, value, any_source, any_tag);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->source, 0);
    EXPECT_EQ(result->tag, 0);
    EXPECT_EQ(result->count, 1u);
    EXPECT_EQ(result->error, 0);
}

}  // namespace dtl::comm_test
