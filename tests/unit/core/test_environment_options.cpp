// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_environment_options.cpp
/// @brief Unit tests for environment_options and backend ownership modes
/// @details Phase 12A: Tests for environment configuration structs and factories.

#include <dtl/core/environment_options.hpp>

#include <gtest/gtest.h>

#include <string_view>

namespace dtl::test {

// =============================================================================
// Default Factory Tests
// =============================================================================

TEST(EnvironmentOptionsTest, DefaultValues) {
    auto opts = environment_options::defaults();

    // MPI defaults: dtl_owns, thread_level FUNNELED, allow fallback
    EXPECT_EQ(opts.mpi.ownership, backend_ownership::dtl_owns);
    EXPECT_EQ(opts.mpi.thread_level, 1);
    EXPECT_TRUE(opts.mpi.allow_thread_fallback);
    EXPECT_EQ(opts.mpi.custom_comm, nullptr);

    // CUDA defaults: optional, auto device
    EXPECT_EQ(opts.cuda.ownership, backend_ownership::optional);
    EXPECT_EQ(opts.cuda.device_id, -1);
    EXPECT_FALSE(opts.cuda.eager_context);

    // SHMEM defaults: disabled
    EXPECT_EQ(opts.shmem.ownership, backend_ownership::disabled);
    EXPECT_EQ(opts.shmem.heap_size, size_type{0});

    // Global defaults
    EXPECT_FALSE(opts.verbose);
}

TEST(EnvironmentOptionsTest, AdoptMpiFactory) {
    auto opts = environment_options::adopt_mpi();

    EXPECT_EQ(opts.mpi.ownership, backend_ownership::adopt_external);

    // Other backends remain at their defaults
    EXPECT_EQ(opts.cuda.ownership, backend_ownership::optional);
    EXPECT_EQ(opts.shmem.ownership, backend_ownership::disabled);
}

TEST(EnvironmentOptionsTest, MpiOnlyFactory) {
    auto opts = environment_options::mpi_only();

    EXPECT_EQ(opts.mpi.ownership, backend_ownership::dtl_owns);
    EXPECT_EQ(opts.cuda.ownership, backend_ownership::disabled);
    EXPECT_EQ(opts.shmem.ownership, backend_ownership::disabled);
}

TEST(EnvironmentOptionsTest, MinimalFactory) {
    auto opts = environment_options::minimal();

    EXPECT_EQ(opts.mpi.ownership, backend_ownership::disabled);
    EXPECT_EQ(opts.cuda.ownership, backend_ownership::disabled);
    EXPECT_EQ(opts.shmem.ownership, backend_ownership::disabled);
}

// =============================================================================
// Backend Ownership Enum Tests
// =============================================================================

TEST(EnvironmentOptionsTest, BackendOwnershipToString) {
    EXPECT_EQ(to_string(backend_ownership::dtl_owns), "dtl_owns");
    EXPECT_EQ(to_string(backend_ownership::adopt_external), "adopt_external");
    EXPECT_EQ(to_string(backend_ownership::optional), "optional");
    EXPECT_EQ(to_string(backend_ownership::disabled), "disabled");
}

// =============================================================================
// Per-Backend Option Defaults
// =============================================================================

TEST(EnvironmentOptionsTest, MpiOptionsDefaults) {
    mpi_options mpi{};

    EXPECT_EQ(mpi.ownership, backend_ownership::dtl_owns);
    EXPECT_EQ(mpi.thread_level, 1);
    EXPECT_TRUE(mpi.allow_thread_fallback);
    EXPECT_EQ(mpi.custom_comm, nullptr);
}

TEST(EnvironmentOptionsTest, CudaOptionsDefaults) {
    cuda_options cuda{};

    EXPECT_EQ(cuda.ownership, backend_ownership::optional);
    EXPECT_EQ(cuda.device_id, -1);
    EXPECT_FALSE(cuda.eager_context);
}

TEST(EnvironmentOptionsTest, ShmemOptionsDefaults) {
    shmem_options shmem{};

    EXPECT_EQ(shmem.ownership, backend_ownership::disabled);
    EXPECT_EQ(shmem.heap_size, size_type{0});
}

// =============================================================================
// Verbose and Custom Comm Tests
// =============================================================================

TEST(EnvironmentOptionsTest, VerboseDefault) {
    environment_options opts{};
    EXPECT_FALSE(opts.verbose);

    // Can be set to true
    opts.verbose = true;
    EXPECT_TRUE(opts.verbose);
}

TEST(EnvironmentOptionsTest, CustomMpiComm) {
    mpi_options mpi{};
    EXPECT_EQ(mpi.custom_comm, nullptr);

    // Set to a non-null value (simulating a cast MPI_Comm pointer)
    int dummy_comm = 42;
    mpi.custom_comm = &dummy_comm;
    EXPECT_NE(mpi.custom_comm, nullptr);
    EXPECT_EQ(mpi.custom_comm, static_cast<void*>(&dummy_comm));
}

}  // namespace dtl::test
