// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_context_with_nccl.cpp
/// @brief Integration tests for context::with_nccl()
/// @details Tests creating NCCL-enabled contexts from MPI contexts.
///          Run with: mpirun -n 2 ./test_context_with_nccl

#include <dtl/core/config.hpp>

// Skip compilation entirely if prerequisites not met
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <dtl/algorithms/context_dispatch.hpp>
#include <dtl/core/context.hpp>
#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>

#include <cuda_runtime.h>
#include <mpi.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace dtl::test {

class ContextWithNcclTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available, skipping NCCL context tests";
        }

        // Get MPI rank
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        device_id_ = rank_ % device_count;
    }

    int device_id_ = 0;
    int rank_ = 0;
    int size_ = 1;
};

TEST_F(ContextWithNcclTest, CreateNcclContextFromMpiContext) {
    // Create base MPI context
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());
    ASSERT_TRUE(ctx.has_mpi());

    // Add NCCL domain
    auto nccl_result = ctx.with_nccl(device_id_);

    ASSERT_TRUE(nccl_result.has_value())
        << "with_nccl should succeed: " << nccl_result.error().message();

    auto& nccl_ctx = *nccl_result;

    // Verify the new context has all expected domains
    EXPECT_TRUE(nccl_ctx.has_mpi()) << "NCCL context should have MPI domain";
    EXPECT_TRUE(nccl_ctx.has_cpu()) << "NCCL context should have CPU domain";
    EXPECT_TRUE(nccl_ctx.has_nccl()) << "NCCL context should have NCCL domain";

    // Verify rank/size are consistent
    EXPECT_EQ(nccl_ctx.rank(), ctx.rank());
    EXPECT_EQ(nccl_ctx.size(), ctx.size());
    EXPECT_EQ(nccl_ctx.is_root(), ctx.is_root());
    EXPECT_TRUE(nccl_ctx.valid());
}

TEST_F(ContextWithNcclTest, NcclContextDomainQueries) {
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());

    auto nccl_result = ctx.with_nccl(device_id_);
    ASSERT_TRUE(nccl_result.has_value());

    auto& nccl_ctx = *nccl_result;

    // Access individual domains
    auto& mpi = nccl_ctx.get<mpi_domain>();
    auto& nccl = nccl_ctx.get<nccl_domain>();

    EXPECT_EQ(mpi.rank(), nccl.rank());
    EXPECT_EQ(mpi.size(), nccl.size());
    EXPECT_TRUE(mpi.valid());
    EXPECT_TRUE(nccl.valid());
}

TEST_F(ContextWithNcclTest, InvalidDeviceIdFails) {
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());

    // Try with invalid device ID
    auto result = ctx.with_nccl(-1);
    EXPECT_TRUE(result.has_error())
        << "with_nccl should fail with invalid device ID";
}

TEST_F(ContextWithNcclTest, AllRanksSucceed) {
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());

    auto nccl_result = ctx.with_nccl(device_id_);
    EXPECT_TRUE(nccl_result.has_value())
        << "All ranks should succeed in creating NCCL context";

    // Barrier to synchronize
    ctx.barrier();

    // Verify all ranks succeeded via MPI reduction
    int local_success = nccl_result.has_value() ? 1 : 0;
    int global_sum = 0;
    MPI_Allreduce(&local_success, &global_sum, 1, MPI_INT, MPI_SUM,
                  ctx.get<mpi_domain>().communicator().underlying().native_handle());

    EXPECT_EQ(global_sum, static_cast<int>(ctx.size()))
        << "All ranks should have succeeded";
}

TEST_F(ContextWithNcclTest, GenericContextDispatchRemainsMpiPrimary) {
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());

    auto nccl_result = ctx.with_nccl(device_id_);
    ASSERT_TRUE(nccl_result.has_value());

    auto& nccl_ctx = *nccl_result;
    auto& comm = dtl::detail::get_comm_adapter(nccl_ctx);

    using actual_comm_t = std::remove_cvref_t<decltype(comm)>;
    using expected_comm_t =
        std::remove_cvref_t<decltype(nccl_ctx.get<mpi_domain>().communicator())>;
    static_assert(std::is_same_v<actual_comm_t, expected_comm_t>,
                  "generic context dispatch must remain MPI-primary");

    EXPECT_EQ(&comm, &nccl_ctx.get<mpi_domain>().communicator());
}

TEST_F(ContextWithNcclTest, SplitNcclCreatesPublicCppSubcontext) {
    mpi_context ctx;
    ASSERT_TRUE(ctx.valid());
    ASSERT_GE(size_, 2);

    auto nccl_result = ctx.with_nccl(device_id_);
    ASSERT_TRUE(nccl_result.has_value());

    auto split_result = nccl_result->split_nccl(rank_ % 2, device_id_);
    ASSERT_TRUE(split_result.has_value())
        << "split_nccl should succeed for explicit C++ NCCL contexts";

    auto& split_ctx = *split_result;
    EXPECT_TRUE(split_ctx.has_mpi());
    EXPECT_TRUE(split_ctx.has_nccl());
    EXPECT_EQ(split_ctx.size(), 1);
    EXPECT_EQ(split_ctx.rank(), 0);
    EXPECT_TRUE(split_ctx.valid());
}

}  // namespace dtl::test

// Custom main for MPI initialization
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}

#else  // !(DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI)

#include <gtest/gtest.h>

TEST(ContextWithNcclTest, SkippedNcclNotEnabled) {
    GTEST_SKIP() << "NCCL, CUDA, or MPI not enabled";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI
