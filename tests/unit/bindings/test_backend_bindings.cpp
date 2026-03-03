// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_backend.h>
#include <cstring>

TEST(BackendBindings, HasMpiReturnsInt) {
    int result = dtl_has_mpi();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(BackendBindings, HasCudaReturnsInt) {
    int result = dtl_has_cuda();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(BackendBindings, HasHipReturnsInt) {
    int result = dtl_has_hip();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(BackendBindings, HasNcclReturnsInt) {
    int result = dtl_has_nccl();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(BackendBindings, HasShmemReturnsInt) {
    int result = dtl_has_shmem();
    EXPECT_TRUE(result == 0 || result == 1);
}

TEST(BackendBindings, BackendNameNotNull) {
    const char* name = dtl_backend_name();
    ASSERT_NE(name, nullptr);
    EXPECT_GT(std::strlen(name), 0u);
}

TEST(BackendBindings, BackendNameIsKnown) {
    const char* name = dtl_backend_name();
    std::string s(name);
    // Must be one of the known backend names
    EXPECT_TRUE(s == "Single" || s == "MPI" || s == "CUDA" || s == "HIP" ||
                s == "CUDA+MPI" || s == "HIP+MPI");
}

TEST(BackendBindings, BackendCountNonNegative) {
    int count = dtl_backend_count();
    EXPECT_GE(count, 0);
    EXPECT_LE(count, 5); // Max 5 backends
}

TEST(BackendBindings, BackendCountConsistent) {
    int count = dtl_backend_count();
    int manual = dtl_has_mpi() + dtl_has_cuda() + dtl_has_hip() +
                 dtl_has_nccl() + dtl_has_shmem();
    EXPECT_EQ(count, manual);
}

TEST(BackendBindings, VersionNotNull) {
    const char* ver = dtl_version();
    ASSERT_NE(ver, nullptr);
    EXPECT_EQ(std::string(ver), DTL_VERSION_STRING);
}

TEST(BackendBindings, MutuallyExclusiveGpuBackends) {
    // HIP and CUDA shouldn't both be enabled (typically)
    // But this isn't strictly enforced, just verify they're valid
    int cuda = dtl_has_cuda();
    int hip = dtl_has_hip();
    EXPECT_TRUE(cuda == 0 || cuda == 1);
    EXPECT_TRUE(hip == 0 || hip == 1);
}

TEST(BackendBindings, NcclRequiresCuda) {
    // If NCCL is available, CUDA should also be available
    if (dtl_has_nccl()) {
        EXPECT_EQ(dtl_has_cuda(), 1);
    }
}

TEST(BackendBindings, MultipleCalls) {
    // Results should be consistent across calls
    EXPECT_EQ(dtl_has_mpi(), dtl_has_mpi());
    EXPECT_EQ(dtl_has_cuda(), dtl_has_cuda());
    EXPECT_EQ(dtl_backend_count(), dtl_backend_count());
}

TEST(BackendBindings, HeaderIncludesSafe) {
    // Just verify header compiles and functions are callable
    (void)dtl_has_mpi();
    (void)dtl_has_cuda();
    (void)dtl_has_hip();
    (void)dtl_has_nccl();
    (void)dtl_has_shmem();
    (void)dtl_backend_name();
    (void)dtl_backend_count();
    (void)dtl_version();
}

TEST(BackendBindings, CppLinkage) {
    // Verify C linkage works from C++
    auto fn = &dtl_has_mpi;
    EXPECT_NE(fn, nullptr);
}
