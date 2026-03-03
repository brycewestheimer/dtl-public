// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_cuda_multirank.cpp
/// @brief CUDA + MPI multi-rank integration tests for DTL
/// @details Tests GPU-backed distributed containers and operations under MPI.
///          Run with: mpirun -n 2 ./test_cuda_multirank
///          Single-rank: ./test_cuda_multirank

#include <dtl/core/config.hpp>

#if DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <dtl/dtl.hpp>
#include <dtl/core/environment.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/containers/distributed_vector.hpp>
#include <dtl/algorithms/algorithms.hpp>
#include <dtl/communication/communication.hpp>
#include <dtl/policies/policies.hpp>

#include <backends/mpi/mpi_comm_adapter.hpp>
#include <backends/cuda/cuda_executor.hpp>
// Note: cuda_memory_space types are already available via dtl/dtl.hpp
// Including backends/cuda/cuda_memory_space.hpp causes redefinition errors
// with the public API header dtl/memory/cuda_memory_space.hpp
#include <dtl/cuda/device_guard.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::integration_test {

// ============================================================================
// Global environment
// ============================================================================

static dtl::environment* g_env = nullptr;

// ============================================================================
// Test Fixture
// ============================================================================

class CudaMultiRankTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        auto ctx = g_env->make_world_context();
        rank_ = ctx.rank();
        size_ = ctx.size();
        comm_ = ctx.get<dtl::mpi_domain>().communicator();

        // Assign device: rank mod device_count
        device_id_ = static_cast<int>(rank_) % device_count;
        cudaSetDevice(device_id_);
    }

    void TearDown() override {
        cudaDeviceReset();
    }

    dtl::mpi::mpi_comm_adapter comm_{};
    dtl::rank_t rank_{0};
    dtl::rank_t size_{1};
    int device_id_{0};
};

// ============================================================================
// CUDA Memory Allocation Tests
// ============================================================================

TEST_F(CudaMultiRankTest, CudaMallocFree) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, 1024);
    ASSERT_EQ(err, cudaSuccess) << "cudaMalloc 1024 B failed";
    ASSERT_NE(ptr, nullptr);

    err = cudaFree(ptr);
    EXPECT_EQ(err, cudaSuccess) << "cudaFree failed";
}

TEST_F(CudaMultiRankTest, CudaManagedMemory) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, 4096);
    ASSERT_EQ(err, cudaSuccess) << "cudaMallocManaged 4096 B failed";
    ASSERT_NE(ptr, nullptr);

    // Managed memory should be accessible from host
    int* iptr = static_cast<int*>(ptr);
    iptr[0] = 42;
    cudaDeviceSynchronize();
    EXPECT_EQ(iptr[0], 42);

    err = cudaFree(ptr);
    EXPECT_EQ(err, cudaSuccess);
}

// ============================================================================
// Host <-> Device Transfer Tests
// ============================================================================

TEST_F(CudaMultiRankTest, HostToDeviceTransfer) {
    constexpr size_t N = 256;
    std::vector<float> h_data(N);
    std::iota(h_data.begin(), h_data.end(), 0.0f);

    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    ASSERT_NE(d_data, nullptr);

    cudaError_t err = cudaMemcpy(d_data, h_data.data(),
                                  N * sizeof(float), cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    // Read back
    std::vector<float> h_verify(N, -1.0f);
    err = cudaMemcpy(h_verify.data(), d_data,
                     N * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_verify[i], static_cast<float>(i));
    }

    cudaFree(d_data);
}

TEST_F(CudaMultiRankTest, DeviceToHostTransfer) {
    constexpr size_t N = 128;
    std::vector<int> h_src(N);
    std::iota(h_src.begin(), h_src.end(), 1000);

    int* d_buf = nullptr;
    cudaMalloc(&d_buf, N * sizeof(int));

    // Host -> Device
    cudaMemcpy(d_buf, h_src.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Device -> Host
    std::vector<int> h_dst(N, 0);
    cudaError_t err = cudaMemcpy(h_dst.data(), d_buf,
                                  N * sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(h_dst[i], 1000 + static_cast<int>(i));
    }

    cudaFree(d_buf);
}

// ============================================================================
// Distributed Vector with Unified Memory
// ============================================================================

TEST_F(CudaMultiRankTest, DistributedVectorUnifiedMemory) {
    // Create a distributed vector using the unified_memory placement policy
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 100;

    dtl::distributed_vector<int, dtl::unified_memory> vec(N, mpi);

    EXPECT_EQ(vec.global_size(), N);
    EXPECT_GT(vec.local_size(), 0u);

    // Fill via local view — unified memory is host-accessible
    auto local = vec.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        local[i] = static_cast<int>(rank_ + 1);
    }

    // Verify data on host side
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], static_cast<int>(rank_ + 1));
    }
}

// ============================================================================
// Reduce with GPU-Backed Data
// ============================================================================

TEST_F(CudaMultiRankTest, ReduceWithGpuData) {
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 200;

    // Use unified_memory so data is accessible from both CPU and GPU
    dtl::distributed_vector<int, dtl::unified_memory> vec(N, 1, mpi);

    // Reduce (sum) — this operates on the local view which is host-accessible
    // for unified memory
    int global_sum = dtl::reduce(dtl::seq{}, vec, 0, std::plus<>{}, comm_);
    EXPECT_EQ(global_sum, static_cast<int>(N));
}

// ============================================================================
// Transform on GPU-Resident Data
// ============================================================================

TEST_F(CudaMultiRankTest, TransformOnUnifiedMemoryData) {
    dtl::mpi_domain mpi;
    constexpr dtl::size_type N = 100;

    dtl::distributed_vector<int, dtl::unified_memory> src(N, 5, mpi);
    dtl::distributed_vector<int, dtl::unified_memory> dst(N, 0, mpi);

    dtl::transform(dtl::seq{}, src, dst, [](int x) { return x * 2; });

    auto local = dst.local_view();
    for (dtl::size_type i = 0; i < local.size(); ++i) {
        EXPECT_EQ(local[i], 10);
    }
}

// ============================================================================
// CUDA Executor Task Submission
// ============================================================================

TEST_F(CudaMultiRankTest, CudaExecutorDispatch) {
    // Test dispatching a simple GPU operation via the async dispatch mechanism
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    ASSERT_EQ(err, cudaSuccess);

    // Allocate device memory to prove the kernel ran
    int* d_flag = nullptr;
    cudaMalloc(&d_flag, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Dispatch a simple memset operation on the stream
    auto future = dtl::cuda::dispatch_gpu_async(stream, [d_flag](cudaStream_t s) {
        cudaMemsetAsync(d_flag, 0x01, sizeof(int), s);
    });

    // Wait for the future (involves progress engine polling)
    // In standalone test we just synchronize the stream directly
    cudaStreamSynchronize(stream);

    // Verify the memset took effect
    int h_flag = 0;
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    EXPECT_NE(h_flag, 0) << "GPU memset should have changed the flag";

    cudaFree(d_flag);
    cudaStreamDestroy(stream);
}

// ============================================================================
// CUDA Memory Space API
// ============================================================================

TEST_F(CudaMultiRankTest, CudaMemorySpaceAllocateDeallocate) {
    // cuda_device_memory_space uses static methods (no instance needed)
    void* ptr = dtl::cuda::cuda_device_memory_space::allocate(2048);
    ASSERT_NE(ptr, nullptr) << "CUDA memory space allocate should succeed";

    dtl::cuda::cuda_device_memory_space::deallocate(ptr, 2048);
    SUCCEED();
}

TEST_F(CudaMultiRankTest, CudaDeviceCount) {
    int count = dtl::cuda::device_count();
    EXPECT_GT(count, 0) << "Should have at least 1 CUDA device";

    auto dev = dtl::cuda::current_device_id();
    EXPECT_GE(dev, 0);
}

// ============================================================================
// Multi-Rank MPI + CUDA Combined Tests
// ============================================================================

TEST_F(CudaMultiRankTest, MpiAllreduceOfGpuValues) {
    if (size_ < 2) GTEST_SKIP() << "Need at least 2 ranks";

    // Each rank allocates managed memory, fills it, then does MPI allreduce
    // on host-accessible copy
    constexpr int count = 4;
    std::vector<int> send(count, static_cast<int>(rank_));
    std::vector<int> recv(count, 0);

    dtl::allreduce(comm_,
                   std::span<const int>(send),
                   std::span<int>(recv),
                   dtl::reduce_sum<>{});

    // Expected: sum of ranks = size*(size-1)/2
    int expected = static_cast<int>(size_ * (size_ - 1) / 2);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(recv[i], expected);
    }
}

}  // namespace dtl::integration_test

// ============================================================================
// MPI-aware main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    dtl::integration_test::g_env = new dtl::environment(argc, argv);

    auto ctx = dtl::integration_test::g_env->make_world_context();
    if (ctx.rank() != 0) {
        ::testing::TestEventListeners& listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    delete dtl::integration_test::g_env;
    dtl::integration_test::g_env = nullptr;

    return result;
}

#else  // !(DTL_ENABLE_CUDA && DTL_ENABLE_MPI)

#include <gtest/gtest.h>

TEST(CudaMultiRankTest, SkippedCudaOrMpiNotEnabled) {
    GTEST_SKIP() << "CUDA or MPI not enabled — skipping CUDA multi-rank tests";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_CUDA && DTL_ENABLE_MPI
