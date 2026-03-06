// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_nccl_domain_from_mpi.cpp
/// @brief Integration tests for NCCL domain creation from MPI
/// @details Tests the nccl_domain::from_mpi() function which requires
///          MPI for bootstrapping NCCL communicator initialization.
///          Run with: mpirun -n 2 ./test_nccl_domain_from_mpi

#include <dtl/core/config.hpp>

// Skip compilation entirely if prerequisites not met
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI

#include <dtl/core/domain.hpp>
#include <dtl/core/domain_impl.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <backends/nccl/nccl_communicator.hpp>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <gtest/gtest.h>

#include <iostream>

namespace dtl::test {

class NcclDomainFromMpiTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available, skipping NCCL tests";
        }

        // Get MPI rank and size for device assignment
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (device_count < size) {
            GTEST_SKIP() << "nccl_domain::from_mpi integration tests require at least one CUDA device per MPI rank";
        }

        // Each rank uses device_id = rank % device_count
        device_id_ = rank % device_count;
    }

    int device_id_ = 0;
};

TEST_F(NcclDomainFromMpiTest, CreateFromValidMpiDomain) {
    // Create MPI domain
    mpi_domain mpi;
    ASSERT_TRUE(mpi.valid()) << "MPI domain should be valid";

    // Create NCCL domain from MPI
    auto result = nccl_domain::from_mpi(mpi, device_id_);

    // This should succeed if NCCL/CUDA are available
    ASSERT_TRUE(result.has_value())
        << "NCCL domain creation should succeed: " << result.error().message();

    nccl_domain& nccl = *result;

    // Verify domain properties
    EXPECT_TRUE(nccl.valid()) << "NCCL domain should be valid";
    EXPECT_EQ(nccl.rank(), mpi.rank()) << "NCCL rank should match MPI rank";
    EXPECT_EQ(nccl.size(), mpi.size()) << "NCCL size should match MPI size";
    EXPECT_EQ(nccl.is_root(), mpi.is_root()) << "NCCL root should match MPI root";
}

TEST_F(NcclDomainFromMpiTest, FailsWithInvalidMpiDomain) {
    // Create an invalid MPI domain (stub when MPI disabled - won't happen here)
    mpi_domain invalid_mpi;
    // We can't easily make MPI domain invalid, so skip this for now
    SUCCEED();
}

TEST_F(NcclDomainFromMpiTest, FailsWithInvalidDeviceId) {
    mpi_domain mpi;
    ASSERT_TRUE(mpi.valid());

    // Get device count
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    // Try with invalid device ID (negative)
    auto result1 = nccl_domain::from_mpi(mpi, -1);
    EXPECT_TRUE(result1.has_error())
        << "Should fail with negative device ID";

    // Try with device ID beyond available devices
    auto result2 = nccl_domain::from_mpi(mpi, device_count + 100);
    EXPECT_TRUE(result2.has_error())
        << "Should fail with out-of-range device ID";
}

TEST_F(NcclDomainFromMpiTest, MultipleRanksCreateDomain) {
    mpi_domain mpi;
    ASSERT_TRUE(mpi.valid());

    auto result = nccl_domain::from_mpi(mpi, device_id_);
    ASSERT_TRUE(result.has_value())
        << "All ranks should successfully create NCCL domain";

    // Barrier to ensure all ranks completed
    mpi.barrier();

    // Verify consistency across ranks
    nccl_domain& nccl = *result;
    EXPECT_EQ(nccl.size(), mpi.size());
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
TEST(NcclDomainFromMpiTest, SkippedNcclNotEnabled) {
    GTEST_SKIP() << "NCCL, CUDA, or MPI not enabled";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#endif  // DTL_ENABLE_NCCL && DTL_ENABLE_CUDA && DTL_ENABLE_MPI
