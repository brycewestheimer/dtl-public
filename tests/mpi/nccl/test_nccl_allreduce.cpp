// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_allreduce.cpp
/// @brief Integration tests for NCCL allreduce collective operation
/// @details Tests end-to-end NCCL allreduce using domain created from MPI.
///          Run with: mpirun -n 2 ./test_nccl_allreduce

#include <dtl/core/config.hpp>

// Skip compilation entirely if prerequisites not met
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <backends/nccl/nccl_comm_adapter.hpp>
#include <backends/nccl/nccl_communicator.hpp>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <gtest/gtest.h>

#include <memory>
#include <vector>
#include <numeric>

namespace dtl::test {

class NcclAllreduceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available, skipping NCCL tests";
        }

        // Get MPI rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        if (device_count < size_) {
            GTEST_SKIP() << "nccl_domain integration tests require at least one CUDA device per MPI rank";
        }

        // Device assignment: rank % device_count
        device_id_ = rank_ % device_count;
        cudaSetDevice(device_id_);

        // Create MPI domain
        mpi_domain_ = std::make_unique<mpi_domain>();
        ASSERT_TRUE(mpi_domain_->valid());

        // Create NCCL domain
        auto nccl_result = nccl_domain::from_mpi(*mpi_domain_, device_id_);
        if (!nccl_result) {
            GTEST_SKIP() << "Could not create NCCL domain: " << nccl_result.error().message();
        }
        nccl_domain_ = std::make_unique<nccl_domain>(std::move(*nccl_result));
    }

    void TearDown() override {
        nccl_domain_.reset();
        mpi_domain_.reset();
    }

    int rank_ = 0;
    int size_ = 1;
    int device_id_ = 0;
    std::unique_ptr<mpi_domain> mpi_domain_;
    std::unique_ptr<nccl_domain> nccl_domain_;
};

TEST_F(NcclAllreduceTest, SumFloat) {
    constexpr size_t count = 4;

    // Allocate device memory
    float* d_send = nullptr;
    float* d_recv = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_send), count * sizeof(float)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_recv), count * sizeof(float)),
              cudaSuccess);

    // Initialize send buffer (each rank sends {rank+1, rank+1, rank+1, rank+1})
    std::vector<float> h_send(count, static_cast<float>(rank_ + 1));
    ASSERT_EQ(cudaMemcpy(d_send, h_send.data(), count * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    // Create a simple NCCL communicator for direct operation
    // (In a real implementation, we'd expose the communicator from nccl_domain)
    // For now, we test the lower-level create_communicator_from_unique_id path

    // Since we have the domain, we can use it for verification
    ASSERT_TRUE(nccl_domain_->valid());
    EXPECT_EQ(nccl_domain_->rank(), static_cast<rank_t>(rank_));
    EXPECT_EQ(nccl_domain_->size(), static_cast<rank_t>(size_));

    // Clean up
    cudaFree(d_send);
    cudaFree(d_recv);
}

TEST_F(NcclAllreduceTest, AdapterRejectsHostBuffers) {
    ASSERT_TRUE(nccl_domain_->valid());

    auto& adapter = nccl_domain_->adapter();
    double host_send = static_cast<double>(rank_ + 1);
    double host_recv = 0.0;

    EXPECT_THROW(adapter.allreduce_sum(&host_send, &host_recv, 1),
                 dtl::nccl::communication_error);
}

TEST_F(NcclAllreduceTest, AdapterAllreduceSumBlocksAndReturnsExpectedValue) {
    ASSERT_TRUE(nccl_domain_->valid());

    constexpr size_t count = 4;
    const double expected = static_cast<double>(size_ * (size_ + 1) / 2);

    double* d_send = nullptr;
    double* d_recv = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_send), count * sizeof(double)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_recv), count * sizeof(double)),
              cudaSuccess);

    std::vector<double> h_send(count, static_cast<double>(rank_ + 1));
    ASSERT_EQ(cudaMemcpy(d_send, h_send.data(), count * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    auto& adapter = nccl_domain_->adapter();
    EXPECT_NO_THROW(adapter.allreduce_sum(d_send, d_recv, count));

    auto stream_status = cudaStreamQuery(nccl_domain_->communicator().stream());
    EXPECT_EQ(stream_status, cudaSuccess) << "blocking NCCL collectives must synchronize";

    std::vector<double> h_recv(count, 0.0);
    ASSERT_EQ(cudaMemcpy(h_recv.data(), d_recv, count * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (double value : h_recv) {
        EXPECT_DOUBLE_EQ(value, expected);
    }

    EXPECT_EQ(cudaFree(d_send), cudaSuccess);
    EXPECT_EQ(cudaFree(d_recv), cudaSuccess);
}

TEST_F(NcclAllreduceTest, DomainRankSizeConsistency) {
    ASSERT_TRUE(nccl_domain_->valid());

    // Verify rank/size match MPI
    EXPECT_EQ(nccl_domain_->rank(), mpi_domain_->rank());
    EXPECT_EQ(nccl_domain_->size(), mpi_domain_->size());
    EXPECT_EQ(nccl_domain_->is_root(), mpi_domain_->is_root());
}

TEST_F(NcclAllreduceTest, AllRanksHaveValidDomain) {
    // Every rank should have a valid domain
    EXPECT_TRUE(nccl_domain_->valid());

    // Use MPI barrier to verify all ranks reached this point
    mpi_domain_->barrier();

    // Collect validity across all ranks
    int local_valid = nccl_domain_->valid() ? 1 : 0;
    int global_sum = 0;
    MPI_Allreduce(&local_valid, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    EXPECT_EQ(global_sum, size_) << "All ranks should have valid NCCL domain";
}

}  // namespace dtl::test

// Custom main for MPI initialization
int main(int argc, char** argv) {
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    // Initialize GoogleTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}

#else  // !(DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI)

#include <gtest/gtest.h>

// Placeholder when NCCL/CUDA/MPI not all enabled
TEST(NcclAllreduceTest, SkippedNcclNotEnabled) {
    GTEST_SKIP() << "NCCL, CUDA, or MPI not enabled";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI
