// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_adapter_contract.cpp
/// @brief Integration tests for the explicit NCCL adapter contract
/// @details Verifies that the public NCCL adapter only accepts device-resident
///          buffers and that blocking collectives synchronize before returning.

#include <dtl/core/config.hpp>

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <dtl/core/context.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#include <backends/nccl/nccl_comm_adapter.hpp>

#include <cuda_runtime.h>
#include <mpi.h>

#include <gtest/gtest.h>

#include <memory>

namespace dtl::test {

using mpi_cpu_nccl_context = context<mpi_domain, cpu_domain, nccl_domain>;

class NcclAdapterContractTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
        if (cuda_err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available, skipping NCCL adapter tests";
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        if (size_ < 2) {
            GTEST_SKIP() << "NCCL adapter tests require at least 2 MPI ranks";
        }

        device_id_ = rank_ % device_count;
        base_ctx_ = std::make_unique<mpi_context>();
        ASSERT_TRUE(base_ctx_->valid());

        auto nccl_result = base_ctx_->with_nccl(device_id_);
        if (!nccl_result) {
            GTEST_SKIP() << "Could not create NCCL context: "
                         << nccl_result.error().message();
        }
        nccl_ctx_ = std::make_unique<mpi_cpu_nccl_context>(std::move(*nccl_result));
    }

    void TearDown() override {
        nccl_ctx_.reset();
        base_ctx_.reset();
    }

    [[nodiscard]] nccl::nccl_comm_adapter& adapter() {
        return nccl_ctx_->get<nccl_domain>().adapter();
    }

    [[nodiscard]] nccl::nccl_communicator& communicator() {
        return nccl_ctx_->get<nccl_domain>().communicator();
    }

    int rank_ = 0;
    int size_ = 1;
    int device_id_ = 0;
    std::unique_ptr<mpi_context> base_ctx_;
    std::unique_ptr<mpi_cpu_nccl_context> nccl_ctx_;
};

TEST_F(NcclAdapterContractTest, RejectsHostBuffersForAllreduceSum) {
    double send = static_cast<double>(rank_ + 1);
    double recv = 0.0;

    EXPECT_THROW(adapter().allreduce_sum(&send, &recv, 1), nccl::communication_error);
}

TEST_F(NcclAdapterContractTest, BlockingAllreduceReturnsCompletedResults) {
    double* d_send = nullptr;
    double* d_recv = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_send), sizeof(double)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&d_recv), sizeof(double)),
              cudaSuccess);

    const double local = static_cast<double>(rank_ + 1);
    ASSERT_EQ(cudaMemcpy(d_send, &local, sizeof(double), cudaMemcpyHostToDevice),
              cudaSuccess);

    ASSERT_NO_THROW(adapter().allreduce_sum(d_send, d_recv, 1));
    EXPECT_EQ(cudaStreamQuery(communicator().stream()), cudaSuccess);

    double global = 0.0;
    ASSERT_EQ(cudaMemcpy(&global, d_recv, sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);

    const double expected = static_cast<double>(size_ * (size_ + 1)) / 2.0;
    EXPECT_DOUBLE_EQ(global, expected);

    EXPECT_EQ(cudaFree(d_send), cudaSuccess);
    EXPECT_EQ(cudaFree(d_recv), cudaSuccess);
}

TEST_F(NcclAdapterContractTest, SplitNcclKeepsExplicitNcclDomainAvailable) {
    auto split_result = nccl_ctx_->split_nccl(/*color=*/0, device_id_);
    ASSERT_TRUE(split_result.has_value())
        << "split_nccl should succeed: " << split_result.error().message();

    auto& split_ctx = *split_result;
    EXPECT_TRUE(split_ctx.has_mpi());
    EXPECT_TRUE(split_ctx.has_nccl());
    EXPECT_EQ(split_ctx.rank(), nccl_ctx_->rank());
    EXPECT_EQ(split_ctx.size(), nccl_ctx_->size());
    EXPECT_TRUE(split_ctx.get<nccl_domain>().valid());
}

}  // namespace dtl::test

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

#else

#include <gtest/gtest.h>

TEST(NcclAdapterContractTest, SkippedNcclNotEnabled) {
    GTEST_SKIP() << "NCCL, CUDA, or MPI not enabled";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI
