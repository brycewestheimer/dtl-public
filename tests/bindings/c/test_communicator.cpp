// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_communicator.cpp
 * @brief Unit tests for DTL C bindings communicator operations
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_communicator.h>
#include <dtl/bindings/c/dtl_context.h>
#include <dtl/bindings/c/dtl_status.h>
#include <vector>
#include <numeric>

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsCommunicator : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
    }

    void TearDown() override {
        if (ctx) {
            dtl_context_destroy(ctx);
        }
    }

    dtl_rank_t rank() { return dtl_context_rank(ctx); }
    dtl_rank_t size() { return dtl_context_size(ctx); }
};

// ============================================================================
// Barrier Tests
// ============================================================================

TEST_F(CBindingsCommunicator, BarrierSucceeds) {
    dtl_status status = dtl_barrier(ctx);
    EXPECT_EQ(status, DTL_SUCCESS);
}

TEST_F(CBindingsCommunicator, BarrierWithNullContextFails) {
    dtl_status status = dtl_barrier(nullptr);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsCommunicator, MultipleBarriersSucceed) {
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(dtl_barrier(ctx), DTL_SUCCESS);
    }
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(CBindingsCommunicator, BroadcastInt32Succeeds) {
    int32_t value = (rank() == 0) ? 42 : 0;

    dtl_status status = dtl_broadcast(ctx, &value, 1, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(value, 42);
}

TEST_F(CBindingsCommunicator, BroadcastFloat64Succeeds) {
    double value = (rank() == 0) ? 3.14159 : 0.0;

    dtl_status status = dtl_broadcast(ctx, &value, 1, DTL_DTYPE_FLOAT64, 0);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(value, 3.14159);
}

TEST_F(CBindingsCommunicator, BroadcastArraySucceeds) {
    std::vector<int32_t> data(10);
    if (rank() == 0) {
        std::iota(data.begin(), data.end(), 1);
    }

    dtl_status status = dtl_broadcast(ctx, data.data(), 10, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(data[i], i + 1);
    }
}

TEST_F(CBindingsCommunicator, BroadcastNullContextFails) {
    int32_t value = 42;
    dtl_status status = dtl_broadcast(nullptr, &value, 1, DTL_DTYPE_INT32, 0);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Reduce Tests
// ============================================================================

TEST_F(CBindingsCommunicator, ReduceSumInt32Succeeds) {
    int32_t sendbuf = rank() + 1;  // rank 0 sends 1, rank 1 sends 2, etc.
    int32_t recvbuf = 0;

    dtl_status status = dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                                    DTL_DTYPE_INT32, DTL_OP_SUM, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    if (rank() == 0) {
        // Sum of 1..size = size * (size + 1) / 2
        int32_t expected = size() * (size() + 1) / 2;
        EXPECT_EQ(recvbuf, expected);
    }
}

TEST_F(CBindingsCommunicator, ReduceMaxFloat64Succeeds) {
    double sendbuf = static_cast<double>(rank() + 1) * 1.5;
    double recvbuf = 0.0;

    dtl_status status = dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                                    DTL_DTYPE_FLOAT64, DTL_OP_MAX, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    if (rank() == 0) {
        double expected = static_cast<double>(size()) * 1.5;
        EXPECT_DOUBLE_EQ(recvbuf, expected);
    }
}

TEST_F(CBindingsCommunicator, ReduceMinInt64Succeeds) {
    int64_t sendbuf = 100 - rank();  // rank 0 sends 100, rank 1 sends 99, etc.
    int64_t recvbuf = 0;

    dtl_status status = dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                                    DTL_DTYPE_INT64, DTL_OP_MIN, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    if (rank() == 0) {
        int64_t expected = 100 - (size() - 1);  // smallest is from highest rank
        EXPECT_EQ(recvbuf, expected);
    }
}

TEST_F(CBindingsCommunicator, ReduceNullContextFails) {
    int32_t sendbuf = 1, recvbuf = 0;
    dtl_status status = dtl_reduce(nullptr, &sendbuf, &recvbuf, 1,
                                    DTL_DTYPE_INT32, DTL_OP_SUM, 0);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// Allreduce Tests
// ============================================================================

TEST_F(CBindingsCommunicator, AllreduceSumInt32Succeeds) {
    int32_t sendbuf = 1;
    int32_t recvbuf = 0;

    dtl_status status = dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                                       DTL_DTYPE_INT32, DTL_OP_SUM);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(recvbuf, size());  // All ranks should have sum
}

TEST_F(CBindingsCommunicator, AllreduceProdFloat32Succeeds) {
    float sendbuf = 2.0f;
    float recvbuf = 0.0f;

    dtl_status status = dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                                       DTL_DTYPE_FLOAT32, DTL_OP_PROD);
    EXPECT_EQ(status, DTL_SUCCESS);

    // 2^size
    float expected = 1.0f;
    for (int i = 0; i < size(); ++i) {
        expected *= 2.0f;
    }
    EXPECT_FLOAT_EQ(recvbuf, expected);
}

TEST_F(CBindingsCommunicator, AllreduceArraySucceeds) {
    std::vector<int32_t> sendbuf(5, rank() + 1);
    std::vector<int32_t> recvbuf(5, 0);

    dtl_status status = dtl_allreduce(ctx, sendbuf.data(), recvbuf.data(), 5,
                                       DTL_DTYPE_INT32, DTL_OP_SUM);
    EXPECT_EQ(status, DTL_SUCCESS);

    int32_t expected = size() * (size() + 1) / 2;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(recvbuf[i], expected);
    }
}

TEST_F(CBindingsCommunicator, HostAllreduceRemainsSafeWithNcclContext) {
    dtl_context_t nccl_ctx = nullptr;
    dtl_status create_status = dtl_context_with_nccl(ctx, /*device_id=*/0, &nccl_ctx);
    if (create_status != DTL_SUCCESS) {
        GTEST_SKIP() << "NCCL context unavailable in this environment";
    }

    int32_t sendbuf = 1;
    int32_t recvbuf = 0;
    dtl_status status = dtl_allreduce(nccl_ctx, &sendbuf, &recvbuf, 1,
                                      DTL_DTYPE_INT32, DTL_OP_SUM);

    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(recvbuf, size());

    dtl_context_destroy(nccl_ctx);
}

// ============================================================================
// Gather Tests
// ============================================================================

TEST_F(CBindingsCommunicator, GatherInt32Succeeds) {
    int32_t sendbuf = rank() + 1;
    std::vector<int32_t> recvbuf(size(), 0);

    dtl_status status = dtl_gather(ctx,
                                    &sendbuf, 1, DTL_DTYPE_INT32,
                                    recvbuf.data(), 1, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    if (rank() == 0) {
        for (int i = 0; i < size(); ++i) {
            EXPECT_EQ(recvbuf[i], i + 1);
        }
    }
}

TEST_F(CBindingsCommunicator, GatherArraySucceeds) {
    std::vector<int32_t> sendbuf(3, rank() + 1);  // Each rank sends 3 elements
    std::vector<int32_t> recvbuf(3 * size(), 0);

    dtl_status status = dtl_gather(ctx,
                                    sendbuf.data(), 3, DTL_DTYPE_INT32,
                                    recvbuf.data(), 3, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    if (rank() == 0) {
        for (int r = 0; r < size(); ++r) {
            for (int i = 0; i < 3; ++i) {
                EXPECT_EQ(recvbuf[r * 3 + i], r + 1);
            }
        }
    }
}

// ============================================================================
// Allgather Tests
// ============================================================================

TEST_F(CBindingsCommunicator, AllgatherInt32Succeeds) {
    int32_t sendbuf = rank() + 1;
    std::vector<int32_t> recvbuf(size(), 0);

    dtl_status status = dtl_allgather(ctx,
                                       &sendbuf, 1, DTL_DTYPE_INT32,
                                       recvbuf.data(), 1, DTL_DTYPE_INT32);
    EXPECT_EQ(status, DTL_SUCCESS);

    // All ranks should have the gathered data
    for (int i = 0; i < size(); ++i) {
        EXPECT_EQ(recvbuf[i], i + 1);
    }
}

// ============================================================================
// Scatter Tests
// ============================================================================

TEST_F(CBindingsCommunicator, ScatterInt32Succeeds) {
    std::vector<int32_t> sendbuf(size());
    if (rank() == 0) {
        std::iota(sendbuf.begin(), sendbuf.end(), 1);
    }

    int32_t recvbuf = 0;

    dtl_status status = dtl_scatter(ctx,
                                     sendbuf.data(), 1, DTL_DTYPE_INT32,
                                     &recvbuf, 1, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);
    EXPECT_EQ(recvbuf, rank() + 1);
}

TEST_F(CBindingsCommunicator, ScatterArraySucceeds) {
    std::vector<int32_t> sendbuf(3 * size());
    if (rank() == 0) {
        for (int r = 0; r < size(); ++r) {
            for (int i = 0; i < 3; ++i) {
                sendbuf[r * 3 + i] = (r + 1) * 10 + i;
            }
        }
    }

    std::vector<int32_t> recvbuf(3, 0);

    dtl_status status = dtl_scatter(ctx,
                                     sendbuf.data(), 3, DTL_DTYPE_INT32,
                                     recvbuf.data(), 3, DTL_DTYPE_INT32, 0);
    EXPECT_EQ(status, DTL_SUCCESS);

    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(recvbuf[i], (rank() + 1) * 10 + i);
    }
}

// ============================================================================
// Alltoall Tests
// ============================================================================

TEST_F(CBindingsCommunicator, AlltoallInt32Succeeds) {
    std::vector<int32_t> sendbuf(size());
    std::vector<int32_t> recvbuf(size(), 0);

    // Each rank sends (rank+1)*10 + dest to destination dest
    for (int i = 0; i < size(); ++i) {
        sendbuf[i] = (rank() + 1) * 10 + i;
    }

    dtl_status status = dtl_alltoall(ctx,
                                      sendbuf.data(), 1, DTL_DTYPE_INT32,
                                      recvbuf.data(), 1, DTL_DTYPE_INT32);
    EXPECT_EQ(status, DTL_SUCCESS);

    // Rank r should receive (src+1)*10 + r from each source
    for (int src = 0; src < size(); ++src) {
        EXPECT_EQ(recvbuf[src], (src + 1) * 10 + rank());
    }
}

// ============================================================================
// Point-to-Point Tests (single process mode)
// ============================================================================

TEST_F(CBindingsCommunicator, SendWithNullContextFails) {
    int32_t buf = 42;
    dtl_status status = dtl_send(nullptr, &buf, 1, DTL_DTYPE_INT32, 0, 0);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsCommunicator, RecvWithNullContextFails) {
    int32_t buf = 0;
    dtl_status status = dtl_recv(nullptr, &buf, 1, DTL_DTYPE_INT32, 0, 0);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsCommunicator, IsendWithNullContextFails) {
    int32_t buf = 42;
    dtl_request_t req;
    dtl_status status = dtl_isend(nullptr, &buf, 1, DTL_DTYPE_INT32, 0, 0, &req);
    EXPECT_NE(status, DTL_SUCCESS);
}

TEST_F(CBindingsCommunicator, IrecvWithNullContextFails) {
    int32_t buf = 0;
    dtl_request_t req;
    dtl_status status = dtl_irecv(nullptr, &buf, 1, DTL_DTYPE_INT32, 0, 0, &req);
    EXPECT_NE(status, DTL_SUCCESS);
}

// ============================================================================
// MPI-specific Point-to-Point Tests (require multiple ranks)
// ============================================================================

#ifdef DTL_HAS_MPI

class CBindingsCommunicatorMPI : public CBindingsCommunicator {};

TEST_F(CBindingsCommunicatorMPI, SendRecvBetweenRanks) {
    if (size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    int32_t sendbuf = rank() * 100;
    int32_t recvbuf = 0;

    if (rank() == 0) {
        // Rank 0 sends to rank 1
        dtl_status status = dtl_send(ctx, &sendbuf, 1, DTL_DTYPE_INT32, 1, 42);
        EXPECT_EQ(status, DTL_SUCCESS);
    } else if (rank() == 1) {
        // Rank 1 receives from rank 0
        dtl_status status = dtl_recv(ctx, &recvbuf, 1, DTL_DTYPE_INT32, 0, 42);
        EXPECT_EQ(status, DTL_SUCCESS);
        EXPECT_EQ(recvbuf, 0);  // Rank 0 sent 0*100 = 0
    }

    dtl_barrier(ctx);
}

TEST_F(CBindingsCommunicatorMPI, SendrecvExchange) {
    if (size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    int32_t sendbuf = rank();
    int32_t recvbuf = -1;

    // Exchange with neighbor
    int partner = (rank() + 1) % 2;  // 0<->1, ignore higher ranks

    if (rank() < 2) {
        dtl_status status = dtl_sendrecv(ctx,
                                          &sendbuf, 1, DTL_DTYPE_INT32, partner, 0,
                                          &recvbuf, 1, DTL_DTYPE_INT32, partner, 0);
        EXPECT_EQ(status, DTL_SUCCESS);
        EXPECT_EQ(recvbuf, partner);
    }

    dtl_barrier(ctx);
}

TEST_F(CBindingsCommunicatorMPI, AsyncSendRecv) {
    if (size() < 2) {
        GTEST_SKIP() << "Test requires at least 2 ranks";
    }

    int32_t sendbuf = rank() * 100;
    int32_t recvbuf = -1;
    dtl_request_t send_req = nullptr;
    dtl_request_t recv_req = nullptr;

    if (rank() == 0) {
        dtl_status status = dtl_isend(ctx, &sendbuf, 1, DTL_DTYPE_INT32, 1, 99, &send_req);
        EXPECT_EQ(status, DTL_SUCCESS);
        EXPECT_NE(send_req, nullptr);

        status = dtl_wait(send_req);
        EXPECT_EQ(status, DTL_SUCCESS);
    } else if (rank() == 1) {
        dtl_status status = dtl_irecv(ctx, &recvbuf, 1, DTL_DTYPE_INT32, 0, 99, &recv_req);
        EXPECT_EQ(status, DTL_SUCCESS);
        EXPECT_NE(recv_req, nullptr);

        status = dtl_wait(recv_req);
        EXPECT_EQ(status, DTL_SUCCESS);
        EXPECT_EQ(recvbuf, 0);
    }

    dtl_barrier(ctx);
}

#endif  // DTL_HAS_MPI
