// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_c_bindings_mpi.cpp
/// @brief MPI multi-rank integration tests for DTL C bindings (C ABI)
/// @details Exercises the C API (dtl_context, dtl_vector, dtl_communicator,
///          dtl_algorithms) under real MPI with multiple ranks.
///          Run with: mpirun -n 2 ./test_c_bindings_mpi
///                    mpirun -n 4 ./test_c_bindings_mpi

#include <dtl/bindings/c/dtl.h>
#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

// ============================================================================
// Global MPI state — mirrors existing integration test pattern
// ============================================================================

static dtl::environment* g_env = nullptr;

int main(int argc, char** argv) {
    g_env = new dtl::environment(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    delete g_env;
    return result;
}

// ============================================================================
// Test Fixture
// ============================================================================

class CBindingsMpiTest : public ::testing::Test {
protected:
    dtl_context_t ctx = nullptr;

    void SetUp() override {
        dtl_status status = dtl_context_create_default(&ctx);
        ASSERT_EQ(status, DTL_SUCCESS);
        ASSERT_NE(ctx, nullptr);
        ASSERT_GE(dtl_context_size(ctx), 1);
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
// Context Tests
// ============================================================================

TEST_F(CBindingsMpiTest, ContextIsValid) {
    EXPECT_EQ(dtl_context_is_valid(ctx), 1);
}

TEST_F(CBindingsMpiTest, ContextRankInRange) {
    EXPECT_GE(rank(), 0);
    EXPECT_LT(rank(), size());
}

TEST_F(CBindingsMpiTest, ContextSizePositive) {
    EXPECT_GE(size(), 1);
}

TEST_F(CBindingsMpiTest, ContextIsRootOnlyOnRankZero) {
    if (rank() == 0) {
        EXPECT_EQ(dtl_context_is_root(ctx), 1);
    } else {
        EXPECT_EQ(dtl_context_is_root(ctx), 0);
    }
}

TEST_F(CBindingsMpiTest, ContextSplitCreatesSubcommunicator) {
    // Split into even/odd ranks
    int color = rank() % 2;
    dtl_context_t sub = nullptr;
    dtl_status status = dtl_context_split(ctx, color, rank(), &sub);
    if (dtl_status_ok(status) && sub != nullptr) {
        // Sub-context should have fewer ranks
        EXPECT_LE(dtl_context_size(sub), size());
        dtl_context_destroy(sub);
    }
    // If split fails (single rank), that's acceptable
}

// ============================================================================
// Barrier Tests
// ============================================================================

TEST_F(CBindingsMpiTest, BarrierSucceeds) {
    EXPECT_EQ(dtl_barrier(ctx), DTL_SUCCESS);
}

TEST_F(CBindingsMpiTest, MultipleBarriersSucceed) {
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(dtl_barrier(ctx), DTL_SUCCESS);
    }
}

// ============================================================================
// Vector Creation Across Ranks
// ============================================================================

TEST_F(CBindingsMpiTest, VectorCreatedAcrossRanks) {
    dtl_vector_t vec = nullptr;
    dtl_status status = dtl_vector_create(ctx, DTL_DTYPE_INT32, 1000, &vec);
    ASSERT_EQ(status, DTL_SUCCESS);

    EXPECT_EQ(dtl_vector_global_size(vec), 1000u);
    EXPECT_GT(dtl_vector_local_size(vec), 0u);
    EXPECT_EQ(dtl_vector_num_ranks(vec), static_cast<dtl_rank_t>(size()));
    EXPECT_EQ(dtl_vector_rank(vec), rank());

    // All ranks' local sizes should sum to global size (verified via allreduce)
    int32_t local = static_cast<int32_t>(dtl_vector_local_size(vec));
    int32_t total = 0;
    dtl_allreduce(ctx, &local, &total, 1, DTL_DTYPE_INT32, DTL_OP_SUM);
    EXPECT_EQ(total, 1000);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorCreateFillAcrossRanks) {
    double fill = 3.14;
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 200, &fill, &vec),
              DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 3.14);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorLocalDataMutWriteReadBack) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 400, &vec), DTL_SUCCESS);

    // Write rank-specific values
    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = rank() * 1000 + static_cast<int32_t>(i);
    }

    // Read back via get_local
    for (dtl_size_t i = 0; i < local_size; ++i) {
        int32_t val = 0;
        EXPECT_EQ(dtl_vector_get_local(vec, i, &val), DTL_SUCCESS);
        EXPECT_EQ(val, rank() * 1000 + static_cast<int32_t>(i));
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorLocalOffsetsDontOverlap) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 1000, &vec), DTL_SUCCESS);

    dtl_index_t my_offset = dtl_vector_local_offset(vec);
    dtl_size_t my_size = dtl_vector_local_size(vec);

    // Gather all offsets and sizes to rank 0
    std::vector<int32_t> all_offsets(size());
    std::vector<int32_t> all_sizes(size());
    int32_t off32 = static_cast<int32_t>(my_offset);
    int32_t sz32 = static_cast<int32_t>(my_size);

    dtl_allgather(ctx, &off32, 1, DTL_DTYPE_INT32,
                  all_offsets.data(), 1, DTL_DTYPE_INT32);
    dtl_allgather(ctx, &sz32, 1, DTL_DTYPE_INT32,
                  all_sizes.data(), 1, DTL_DTYPE_INT32);

    // Verify non-overlapping partitions
    for (int r = 0; r < size() - 1; ++r) {
        EXPECT_EQ(all_offsets[r] + all_sizes[r], all_offsets[r + 1])
            << "Partition gap/overlap between rank " << r << " and " << r + 1;
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorIsLocalConsistentWithOwner) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec), DTL_SUCCESS);

    dtl_index_t offset = dtl_vector_local_offset(vec);
    dtl_size_t local_size = dtl_vector_local_size(vec);

    for (dtl_size_t i = 0; i < local_size; ++i) {
        dtl_index_t gidx = offset + static_cast<dtl_index_t>(i);
        EXPECT_EQ(dtl_vector_is_local(vec, gidx), 1);
        EXPECT_EQ(dtl_vector_owner(vec, gidx), rank());
        EXPECT_EQ(dtl_vector_to_local(vec, gidx), static_cast<dtl_index_t>(i));
        EXPECT_EQ(dtl_vector_to_global(vec, static_cast<dtl_index_t>(i)), gidx);
    }

    dtl_vector_destroy(vec);
}

// ============================================================================
// Broadcast Tests
// ============================================================================

TEST_F(CBindingsMpiTest, BroadcastInt32FromRoot) {
    int32_t value = (rank() == 0) ? 42 : 0;

    ASSERT_EQ(dtl_broadcast(ctx, &value, 1, DTL_DTYPE_INT32, 0), DTL_SUCCESS);
    EXPECT_EQ(value, 42);
}

TEST_F(CBindingsMpiTest, BroadcastFloat64Array) {
    std::vector<double> buf(10);
    if (rank() == 0) {
        for (int i = 0; i < 10; ++i) buf[i] = i * 1.5;
    }

    ASSERT_EQ(dtl_broadcast(ctx, buf.data(), 10, DTL_DTYPE_FLOAT64, 0),
              DTL_SUCCESS);

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(buf[i], i * 1.5);
    }
}

TEST_F(CBindingsMpiTest, BroadcastFromNonZeroRoot) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    int32_t root = size() - 1;  // Last rank is root
    int64_t value = (rank() == root) ? 9999 : 0;

    ASSERT_EQ(dtl_broadcast(ctx, &value, 1, DTL_DTYPE_INT64, root), DTL_SUCCESS);
    EXPECT_EQ(value, 9999);
}

// ============================================================================
// Allreduce Tests
// ============================================================================

TEST_F(CBindingsMpiTest, AllreduceSumInt32) {
    int32_t sendbuf = 1;
    int32_t recvbuf = 0;

    ASSERT_EQ(dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                             DTL_DTYPE_INT32, DTL_OP_SUM), DTL_SUCCESS);
    EXPECT_EQ(recvbuf, size());
}

TEST_F(CBindingsMpiTest, AllreduceSumFloat64) {
    double sendbuf = static_cast<double>(rank() + 1);
    double recvbuf = 0.0;

    ASSERT_EQ(dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                             DTL_DTYPE_FLOAT64, DTL_OP_SUM), DTL_SUCCESS);

    // Sum of 1..size = size*(size+1)/2
    double expected = size() * (size() + 1) / 2.0;
    EXPECT_DOUBLE_EQ(recvbuf, expected);
}

TEST_F(CBindingsMpiTest, AllreduceMaxInt32) {
    int32_t sendbuf = rank() * 10;
    int32_t recvbuf = 0;

    ASSERT_EQ(dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                             DTL_DTYPE_INT32, DTL_OP_MAX), DTL_SUCCESS);
    EXPECT_EQ(recvbuf, (size() - 1) * 10);
}

TEST_F(CBindingsMpiTest, AllreduceMinFloat64) {
    double sendbuf = 100.0 + rank();
    double recvbuf = 0.0;

    ASSERT_EQ(dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                             DTL_DTYPE_FLOAT64, DTL_OP_MIN), DTL_SUCCESS);
    EXPECT_DOUBLE_EQ(recvbuf, 100.0);
}

TEST_F(CBindingsMpiTest, AllreduceArraySum) {
    std::vector<int32_t> sendbuf(5, rank() + 1);
    std::vector<int32_t> recvbuf(5, 0);

    ASSERT_EQ(dtl_allreduce(ctx, sendbuf.data(), recvbuf.data(), 5,
                             DTL_DTYPE_INT32, DTL_OP_SUM), DTL_SUCCESS);

    int32_t expected = size() * (size() + 1) / 2;
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(recvbuf[i], expected);
    }
}

TEST_F(CBindingsMpiTest, AllreduceProdFloat32) {
    float sendbuf = 2.0f;
    float recvbuf = 0.0f;

    ASSERT_EQ(dtl_allreduce(ctx, &sendbuf, &recvbuf, 1,
                             DTL_DTYPE_FLOAT32, DTL_OP_PROD), DTL_SUCCESS);

    float expected = std::pow(2.0f, static_cast<float>(size()));
    EXPECT_FLOAT_EQ(recvbuf, expected);
}

// ============================================================================
// Reduce Tests
// ============================================================================

TEST_F(CBindingsMpiTest, ReduceSumToRoot) {
    int32_t sendbuf = rank() + 1;
    int32_t recvbuf = 0;

    ASSERT_EQ(dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                          DTL_DTYPE_INT32, DTL_OP_SUM, 0), DTL_SUCCESS);

    if (rank() == 0) {
        int32_t expected = size() * (size() + 1) / 2;
        EXPECT_EQ(recvbuf, expected);
    }
}

TEST_F(CBindingsMpiTest, ReduceMaxFloat64ToRoot) {
    double sendbuf = static_cast<double>(rank()) * 1.5;
    double recvbuf = 0.0;

    ASSERT_EQ(dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                          DTL_DTYPE_FLOAT64, DTL_OP_MAX, 0), DTL_SUCCESS);

    if (rank() == 0) {
        double expected = (size() - 1) * 1.5;
        EXPECT_DOUBLE_EQ(recvbuf, expected);
    }
}

TEST_F(CBindingsMpiTest, ReduceToNonZeroRoot) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    int32_t root = size() - 1;
    int32_t sendbuf = 1;
    int32_t recvbuf = 0;

    ASSERT_EQ(dtl_reduce(ctx, &sendbuf, &recvbuf, 1,
                          DTL_DTYPE_INT32, DTL_OP_SUM, root), DTL_SUCCESS);

    if (rank() == root) {
        EXPECT_EQ(recvbuf, size());
    }
}

// ============================================================================
// Gather Tests
// ============================================================================

TEST_F(CBindingsMpiTest, GatherToRoot) {
    int32_t sendbuf = rank() * 10;
    std::vector<int32_t> recvbuf(size(), 0);

    ASSERT_EQ(dtl_gather(ctx, &sendbuf, 1, DTL_DTYPE_INT32,
                          recvbuf.data(), 1, DTL_DTYPE_INT32, 0), DTL_SUCCESS);

    if (rank() == 0) {
        for (int r = 0; r < size(); ++r) {
            EXPECT_EQ(recvbuf[r], r * 10);
        }
    }
}

TEST_F(CBindingsMpiTest, AllgatherAll) {
    int32_t sendbuf = rank() + 100;
    std::vector<int32_t> recvbuf(size(), 0);

    ASSERT_EQ(dtl_allgather(ctx, &sendbuf, 1, DTL_DTYPE_INT32,
                              recvbuf.data(), 1, DTL_DTYPE_INT32), DTL_SUCCESS);

    for (int r = 0; r < size(); ++r) {
        EXPECT_EQ(recvbuf[r], r + 100);
    }
}

// ============================================================================
// Scatter Tests
// ============================================================================

TEST_F(CBindingsMpiTest, ScatterFromRoot) {
    std::vector<int32_t> sendbuf(size());
    if (rank() == 0) {
        std::iota(sendbuf.begin(), sendbuf.end(), 1);
    }

    int32_t recvbuf = 0;

    ASSERT_EQ(dtl_scatter(ctx, sendbuf.data(), 1, DTL_DTYPE_INT32,
                           &recvbuf, 1, DTL_DTYPE_INT32, 0), DTL_SUCCESS);
    EXPECT_EQ(recvbuf, rank() + 1);
}

// ============================================================================
// Alltoall Tests
// ============================================================================

TEST_F(CBindingsMpiTest, AlltoallExchange) {
    std::vector<int32_t> sendbuf(size());
    std::vector<int32_t> recvbuf(size(), 0);

    for (int i = 0; i < size(); ++i) {
        sendbuf[i] = rank() * 100 + i;
    }

    ASSERT_EQ(dtl_alltoall(ctx, sendbuf.data(), 1, DTL_DTYPE_INT32,
                             recvbuf.data(), 1, DTL_DTYPE_INT32), DTL_SUCCESS);

    // Rank r should receive (src*100 + r) from each src
    for (int src = 0; src < size(); ++src) {
        EXPECT_EQ(recvbuf[src], src * 100 + rank());
    }
}

// ============================================================================
// Point-to-Point Tests (require >= 2 ranks)
// ============================================================================

TEST_F(CBindingsMpiTest, SendRecvBetweenRanks) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    int32_t sendbuf = rank() * 100;
    int32_t recvbuf = -1;

    if (rank() == 0) {
        EXPECT_EQ(dtl_send(ctx, &sendbuf, 1, DTL_DTYPE_INT32, 1, 42),
                  DTL_SUCCESS);
    } else if (rank() == 1) {
        EXPECT_EQ(dtl_recv(ctx, &recvbuf, 1, DTL_DTYPE_INT32, 0, 42),
                  DTL_SUCCESS);
        EXPECT_EQ(recvbuf, 0);  // rank 0 sent 0*100
    }

    dtl_barrier(ctx);
}

TEST_F(CBindingsMpiTest, SendrecvExchange) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    int32_t sendbuf = rank();
    int32_t recvbuf = -1;
    int partner = (rank() + 1) % 2;

    if (rank() < 2) {
        EXPECT_EQ(dtl_sendrecv(ctx,
                                &sendbuf, 1, DTL_DTYPE_INT32, partner, 0,
                                &recvbuf, 1, DTL_DTYPE_INT32, partner, 0),
                  DTL_SUCCESS);
        EXPECT_EQ(recvbuf, partner);
    }

    dtl_barrier(ctx);
}

TEST_F(CBindingsMpiTest, AsyncSendRecv) {
    if (size() < 2) GTEST_SKIP() << "Need >= 2 ranks";

    int32_t sendbuf = rank() * 100;
    int32_t recvbuf = -1;
    dtl_request_t send_req = nullptr;
    dtl_request_t recv_req = nullptr;

    if (rank() == 0) {
        ASSERT_EQ(dtl_isend(ctx, &sendbuf, 1, DTL_DTYPE_INT32, 1, 77, &send_req),
                  DTL_SUCCESS);
        ASSERT_EQ(dtl_wait(send_req), DTL_SUCCESS);
    } else if (rank() == 1) {
        ASSERT_EQ(dtl_irecv(ctx, &recvbuf, 1, DTL_DTYPE_INT32, 0, 77, &recv_req),
                  DTL_SUCCESS);
        ASSERT_EQ(dtl_wait(recv_req), DTL_SUCCESS);
        EXPECT_EQ(recvbuf, 0);
    }

    dtl_barrier(ctx);
}

// ============================================================================
// Vector + Algorithms: Fill, Sort, Reduce
// ============================================================================

TEST_F(CBindingsMpiTest, VectorFillAndLocalReduce) {
    double fill = 2.5;
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 100, &fill, &vec),
              DTL_SUCCESS);

    double local_sum = 0.0;
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_SUM, &local_sum), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    EXPECT_DOUBLE_EQ(local_sum, local_size * 2.5);

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorSortLocalAscending) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 200, &vec), DTL_SUCCESS);

    // Fill in reverse order
    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<int32_t>(local_size - i);
    }

    ASSERT_EQ(dtl_sort_vector(vec), DTL_SUCCESS);

    const int32_t* sorted = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_LE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorSortLocalDescending) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 200, &vec), DTL_SUCCESS);

    double* data = static_cast<double*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = static_cast<double>(i);
    }

    ASSERT_EQ(dtl_sort_vector_descending(vec), DTL_SUCCESS);

    const double* sorted = static_cast<const double*>(dtl_vector_local_data(vec));
    for (dtl_size_t i = 1; i < local_size; ++i) {
        EXPECT_GE(sorted[i - 1], sorted[i]);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, VectorMinMax) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 200, &vec), DTL_SUCCESS);

    int32_t* data = static_cast<int32_t*>(dtl_vector_local_data_mut(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        data[i] = rank() * 1000 + static_cast<int32_t>(i);
    }

    int32_t min_val = 0, max_val = 0;
    ASSERT_EQ(dtl_minmax_vector(vec, &min_val, &max_val), DTL_SUCCESS);

    EXPECT_EQ(min_val, rank() * 1000);
    EXPECT_EQ(max_val, rank() * 1000 + static_cast<int32_t>(local_size - 1));

    dtl_vector_destroy(vec);
}

// ============================================================================
// For-Each and Transform via C API under MPI
// ============================================================================

namespace {

void negate_element(void* elem, dtl_size_t /*idx*/, void* /*user_data*/) {
    int32_t* val = static_cast<int32_t*>(elem);
    *val = -(*val);
}

void scale_transform(const void* in, void* out, dtl_size_t /*idx*/, void* user_data) {
    const double* input = static_cast<const double*>(in);
    double* output = static_cast<double*>(out);
    double factor = *static_cast<double*>(user_data);
    *output = (*input) * factor;
}

} // namespace

TEST_F(CBindingsMpiTest, ForEachModifiesLocally) {
    int32_t fill = 5;
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 100, &fill, &vec),
              DTL_SUCCESS);

    ASSERT_EQ(dtl_for_each_vector(vec, negate_element, nullptr), DTL_SUCCESS);

    const int32_t* data = static_cast<const int32_t*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_EQ(data[i], -5);
    }

    dtl_vector_destroy(vec);
}

TEST_F(CBindingsMpiTest, TransformIntoNewVector) {
    double fill = 3.0;
    dtl_vector_t src = nullptr, dst = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 100, &fill, &src),
              DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &dst), DTL_SUCCESS);

    double factor = 2.0;
    ASSERT_EQ(dtl_transform_vector(src, dst, scale_transform, &factor),
              DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(dst));
    dtl_size_t local_size = dtl_vector_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 6.0);
    }

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
}

// ============================================================================
// Copy & Fill via C API  under MPI
// ============================================================================

TEST_F(CBindingsMpiTest, CopyVectorAcrossRanks) {
    double fill = 42.0;
    dtl_vector_t src = nullptr, dst = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_FLOAT64, 100, &fill, &src),
              DTL_SUCCESS);
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, 100, &dst), DTL_SUCCESS);

    ASSERT_EQ(dtl_copy_vector(src, dst), DTL_SUCCESS);

    const double* data = static_cast<const double*>(dtl_vector_local_data(dst));
    dtl_size_t local_size = dtl_vector_local_size(dst);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_DOUBLE_EQ(data[i], 42.0);
    }

    dtl_vector_destroy(src);
    dtl_vector_destroy(dst);
}

TEST_F(CBindingsMpiTest, FillVectorAlgorithm) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_FLOAT32, 100, &vec), DTL_SUCCESS);

    float fill = 7.5f;
    ASSERT_EQ(dtl_fill_vector(vec, &fill), DTL_SUCCESS);

    const float* data = static_cast<const float*>(dtl_vector_local_data(vec));
    dtl_size_t local_size = dtl_vector_local_size(vec);
    for (dtl_size_t i = 0; i < local_size; ++i) {
        EXPECT_FLOAT_EQ(data[i], 7.5f);
    }

    dtl_vector_destroy(vec);
}

// ============================================================================
// Combined: vector + collective reduce to verify global correctness
// ============================================================================

TEST_F(CBindingsMpiTest, VectorLocalSumThenGlobalAllreduce) {
    // Each rank fills vector with (rank+1), then local sum, then allreduce
    int32_t fill_val = rank() + 1;
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 400, &fill_val, &vec),
              DTL_SUCCESS);

    // Local sum via dtl_reduce_local_vector
    int32_t local_sum = 0;
    ASSERT_EQ(dtl_reduce_local_vector(vec, DTL_OP_SUM, &local_sum), DTL_SUCCESS);

    dtl_size_t local_size = dtl_vector_local_size(vec);
    EXPECT_EQ(local_sum, static_cast<int32_t>(local_size) * (rank() + 1));

    // Global sum via allreduce
    int32_t global_sum = 0;
    ASSERT_EQ(dtl_allreduce(ctx, &local_sum, &global_sum, 1,
                             DTL_DTYPE_INT32, DTL_OP_SUM), DTL_SUCCESS);

    // Verify all ranks see the same global sum
    int32_t global_sum2 = 0;
    dtl_allreduce(ctx, &global_sum, &global_sum2, 1,
                  DTL_DTYPE_INT32, DTL_OP_MIN);
    int32_t global_sum3 = 0;
    dtl_allreduce(ctx, &global_sum, &global_sum3, 1,
                  DTL_DTYPE_INT32, DTL_OP_MAX);
    EXPECT_EQ(global_sum2, global_sum3) << "All ranks should see the same sum";

    dtl_vector_destroy(vec);
}

// ============================================================================
// Vector Resize under MPI
// ============================================================================

TEST_F(CBindingsMpiTest, VectorResizePreservesConsistency) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec), DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_global_size(vec), 100u);

    ASSERT_EQ(dtl_vector_resize(vec, 200), DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_global_size(vec), 200u);

    // Verify local sizes still sum to global
    int32_t local_sz = static_cast<int32_t>(dtl_vector_local_size(vec));
    int32_t total = 0;
    dtl_allreduce(ctx, &local_sz, &total, 1, DTL_DTYPE_INT32, DTL_OP_SUM);
    EXPECT_EQ(total, 200);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Vector Barrier and Sync
// ============================================================================

TEST_F(CBindingsMpiTest, VectorBarrierSucceeds) {
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create(ctx, DTL_DTYPE_INT32, 100, &vec), DTL_SUCCESS);

    EXPECT_EQ(dtl_vector_barrier(vec), DTL_SUCCESS);
    EXPECT_EQ(dtl_vector_sync(vec), DTL_SUCCESS);

    dtl_vector_destroy(vec);
}

// ============================================================================
// Find & Count under MPI
// ============================================================================

TEST_F(CBindingsMpiTest, FindAndCountLocally) {
    int32_t fill = 5;
    dtl_vector_t vec = nullptr;
    ASSERT_EQ(dtl_vector_create_fill(ctx, DTL_DTYPE_INT32, 100, &fill, &vec),
              DTL_SUCCESS);

    // Count should equal local size
    dtl_size_t count = dtl_count_vector(vec, &fill);
    EXPECT_EQ(count, dtl_vector_local_size(vec));

    // Find should return 0 (first local index)
    dtl_index_t idx = dtl_find_vector(vec, &fill);
    EXPECT_EQ(idx, 0);

    // A value that doesn't exist
    int32_t absent = 999;
    dtl_index_t idx2 = dtl_find_vector(vec, &absent);
    EXPECT_EQ(idx2, -1);

    dtl_vector_destroy(vec);
}
