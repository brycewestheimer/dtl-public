// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_multirank.cpp
/// @brief NCCL + CUDA + MPI multi-rank integration tests for DTL
/// @details Tests NCCL collective operations (allreduce, broadcast, reduce,
///          reduce-scatter, barrier) on GPU-resident buffers using raw NCCL
///          API with MPI for rank coordination.
///          Run with: mpirun -n 2 ./test_nccl_multirank
///
/// @note The DTL nccl_communicator wrapper has pre-existing API
///       incompatibilities with the current communicator_base class.
///       These tests exercise NCCL functionality directly via the raw
///       NCCL C API to validate hardware + driver correctness.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace dtl::integration_test {

class NcclMultiRankTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        if (size_ < 2) {
            GTEST_SKIP() << "NCCL tests require at least 2 MPI ranks";
        }

        device_id_ = rank_ % device_count;
        cudaSetDevice(device_id_);

        err = cudaStreamCreate(&stream_);
        ASSERT_EQ(err, cudaSuccess) << "cudaStreamCreate failed";

        ncclUniqueId nccl_id;
        if (rank_ == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

        ncclResult_t nccl_err = ncclCommInitRank(&comm_, size_, nccl_id, rank_);
        if (nccl_err != ncclSuccess) {
            GTEST_SKIP() << "NCCL init failed: " << ncclGetErrorString(nccl_err);
        }
        comm_valid_ = true;
    }

    void TearDown() override {
        if (comm_valid_) {
            ncclCommDestroy(comm_);
            comm_valid_ = false;
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    int rank_ = 0;
    int size_ = 1;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    ncclComm_t comm_ = nullptr;
    bool comm_valid_ = false;
};

TEST_F(NcclMultiRankTest, CommCreation) {
    ASSERT_TRUE(comm_valid_);
    int nccl_count = 0;
    ncclCommCount(comm_, &nccl_count);
    EXPECT_EQ(nccl_count, size_);

    int nccl_rank = -1;
    ncclCommUserRank(comm_, &nccl_rank);
    EXPECT_EQ(nccl_rank, rank_);
}

TEST_F(NcclMultiRankTest, AllRanksHaveValidComm) {
    int local_valid = comm_valid_ ? 1 : 0;
    int global_sum = 0;
    MPI_Allreduce(&local_valid, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(global_sum, size_);
}

TEST_F(NcclMultiRankTest, AllreduceSumFloat) {
    constexpr size_t count = 8;

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaMalloc(&d_send, count * sizeof(float));
    cudaMalloc(&d_recv, count * sizeof(float));

    std::vector<float> h_send(count, static_cast<float>(rank_ + 1));
    cudaMemcpy(d_send, h_send.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess) << ncclGetErrorString(res);

    std::vector<float> h_recv(count, 0.0f);
    cudaMemcpy(h_recv.data(), d_recv, count * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = static_cast<float>(size_ * (size_ + 1)) / 2.0f;
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_recv[i], expected) << "index=" << i;
    }

    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclMultiRankTest, AllreduceSumInt) {
    constexpr size_t count = 4;

    int* d_send = nullptr;
    int* d_recv = nullptr;
    cudaMalloc(&d_send, count * sizeof(int));
    cudaMalloc(&d_recv, count * sizeof(int));

    std::vector<int> h_send(count, rank_ + 1);
    cudaMemcpy(d_send, h_send.data(), count * sizeof(int), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclAllReduce(d_send, d_recv, count, ncclInt, ncclSum,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    std::vector<int> h_recv(count, 0);
    cudaMemcpy(h_recv.data(), d_recv, count * sizeof(int), cudaMemcpyDeviceToHost);

    int expected = size_ * (size_ + 1) / 2;
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(h_recv[i], expected) << "index=" << i;
    }

    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclMultiRankTest, AllreduceInplace) {
    constexpr size_t count = 4;

    float* d_buf = nullptr;
    cudaMalloc(&d_buf, count * sizeof(float));

    std::vector<float> h_data(count, static_cast<float>(rank_ + 1));
    cudaMemcpy(d_buf, h_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    std::vector<float> h_result(count, 0.0f);
    cudaMemcpy(h_result.data(), d_buf, count * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = static_cast<float>(size_ * (size_ + 1)) / 2.0f;
    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], expected);
    }

    cudaFree(d_buf);
}

TEST_F(NcclMultiRankTest, AllreduceLargeBuffer) {
    constexpr size_t count = 1024 * 1024;

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaMalloc(&d_send, count * sizeof(float));
    cudaMalloc(&d_recv, count * sizeof(float));

    float rank_f = static_cast<float>(rank_);
    std::vector<float> h_send(count, rank_f);
    cudaMemcpy(d_send, h_send.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclAllReduce(d_send, d_recv, count, ncclFloat, ncclSum,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    std::vector<float> h_recv(16, 0.0f);
    cudaMemcpy(h_recv.data(), d_recv, 16 * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = static_cast<float>(size_ * (size_ - 1)) / 2.0f;
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(h_recv[i], expected);
    }

    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclMultiRankTest, BroadcastFromRoot) {
    constexpr size_t count = 16;
    constexpr int root = 0;

    float* d_buf = nullptr;
    cudaMalloc(&d_buf, count * sizeof(float));

    std::vector<float> h_data(count);
    if (rank_ == root) {
        std::iota(h_data.begin(), h_data.end(), 100.0f);
    } else {
        std::fill(h_data.begin(), h_data.end(), -1.0f);
    }
    cudaMemcpy(d_buf, h_data.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclBroadcast(d_buf, d_buf, count, ncclFloat, root,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    std::vector<float> h_result(count, 0.0f);
    cudaMemcpy(h_result.data(), d_buf, count * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 100.0f + static_cast<float>(i))
            << "rank=" << rank_ << " index=" << i;
    }

    cudaFree(d_buf);
}

TEST_F(NcclMultiRankTest, ReduceToRoot) {
    constexpr size_t count = 4;
    constexpr int root = 0;

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaMalloc(&d_send, count * sizeof(float));
    cudaMalloc(&d_recv, count * sizeof(float));

    std::vector<float> h_send(count, static_cast<float>(rank_));
    cudaMemcpy(d_send, h_send.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclReduce(d_send, d_recv, count, ncclFloat, ncclSum,
                                   root, comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    if (rank_ == root) {
        std::vector<float> h_recv(count, 0.0f);
        cudaMemcpy(h_recv.data(), d_recv, count * sizeof(float), cudaMemcpyDeviceToHost);

        float expected = static_cast<float>(size_ * (size_ - 1)) / 2.0f;
        for (size_t i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(h_recv[i], expected) << "index=" << i;
        }
    }

    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclMultiRankTest, ReduceScatter) {
    size_t recv_count = 1;
    size_t send_count = static_cast<size_t>(size_);

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaMalloc(&d_send, send_count * sizeof(float));
    cudaMalloc(&d_recv, recv_count * sizeof(float));

    std::vector<float> h_send(send_count);
    for (size_t i = 0; i < send_count; ++i) {
        h_send[i] = static_cast<float>(rank_ + static_cast<int>(i));
    }
    cudaMemcpy(d_send, h_send.data(), send_count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclReduceScatter(d_send, d_recv, recv_count, ncclFloat,
                                          ncclSum, comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    float h_recv = 0.0f;
    cudaMemcpy(&h_recv, d_recv, sizeof(float), cudaMemcpyDeviceToHost);

    float expected = static_cast<float>(size_ * (size_ - 1) / 2 + size_ * rank_);
    EXPECT_FLOAT_EQ(h_recv, expected);

    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclMultiRankTest, AllGather) {
    constexpr size_t send_count = 2;
    size_t recv_count = send_count * static_cast<size_t>(size_);

    float* d_send = nullptr;
    float* d_recv = nullptr;
    cudaMalloc(&d_send, send_count * sizeof(float));
    cudaMalloc(&d_recv, recv_count * sizeof(float));

    std::vector<float> h_send = {static_cast<float>(rank_ * 10),
                                  static_cast<float>(rank_ * 10 + 1)};
    cudaMemcpy(d_send, h_send.data(), send_count * sizeof(float), cudaMemcpyHostToDevice);

    ncclResult_t res = ncclAllGather(d_send, d_recv, send_count, ncclFloat,
                                      comm_, stream_);
    cudaStreamSynchronize(stream_);
    ASSERT_EQ(res, ncclSuccess);

    std::vector<float> h_recv(recv_count, 0.0f);
    cudaMemcpy(h_recv.data(), d_recv, recv_count * sizeof(float), cudaMemcpyDeviceToHost);

    for (int r = 0; r < size_; ++r) {
        EXPECT_FLOAT_EQ(h_recv[r * 2], static_cast<float>(r * 10));
        EXPECT_FLOAT_EQ(h_recv[r * 2 + 1], static_cast<float>(r * 10 + 1));
    }

    cudaFree(d_send);
    cudaFree(d_recv);
}

}  // namespace dtl::integration_test

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

#else

#include <gtest/gtest.h>

TEST(NcclMultiRankTest, SkippedNcclNotEnabled) {
    GTEST_SKIP() << "NCCL, CUDA, or MPI not enabled";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif
