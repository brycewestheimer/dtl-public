// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_backend_concept_conformance.cpp
/// @brief Unit tests for Phase 07 - Backend Concept Conformance
/// @details Tests concept checks, backend_traits, topology, and event fixes.

#include <dtl/backend/backend.hpp>
#include <dtl/backend/common/backend_traits.hpp>
#include <dtl/backend/concepts/topology.hpp>
#include <dtl/backend/concepts/executor.hpp>
#include <dtl/backend/concepts/event.hpp>
#include <dtl/backend/concepts/memory_space.hpp>

#include <gtest/gtest.h>

#include <string>
#include <thread>

namespace dtl::test {

// =============================================================================
// Backend Traits Tests (T03)
// =============================================================================

TEST(BackendTraitsTest, MpiTraitsExist) {
    EXPECT_TRUE(backend_traits<mpi_backend_tag>::supports_collectives);
    EXPECT_TRUE(backend_traits<mpi_backend_tag>::supports_point_to_point);
    EXPECT_TRUE(backend_traits<mpi_backend_tag>::supports_rma);
    EXPECT_STREQ(backend_traits<mpi_backend_tag>::name, "MPI");
}

TEST(BackendTraitsTest, CudaTraitsExist) {
    EXPECT_TRUE(backend_traits<cuda_backend_tag>::supports_gpu_aware);
    EXPECT_TRUE(backend_traits<cuda_backend_tag>::supports_async);
    EXPECT_STREQ(backend_traits<cuda_backend_tag>::name, "CUDA");
}

TEST(BackendTraitsTest, HipTraitsExist) {
    EXPECT_TRUE(backend_traits<hip_backend_tag>::supports_gpu_aware);
    EXPECT_TRUE(backend_traits<hip_backend_tag>::supports_async);
    EXPECT_FALSE(backend_traits<hip_backend_tag>::supports_rma);
    EXPECT_STREQ(backend_traits<hip_backend_tag>::name, "HIP");
}

TEST(BackendTraitsTest, ShmemTraitsExist) {
    EXPECT_TRUE(backend_traits<shmem_backend_tag>::supports_rma);
    EXPECT_TRUE(backend_traits<shmem_backend_tag>::supports_point_to_point);
    EXPECT_TRUE(backend_traits<shmem_backend_tag>::supports_collectives);
    EXPECT_FALSE(backend_traits<shmem_backend_tag>::supports_gpu_aware);
    EXPECT_STREQ(backend_traits<shmem_backend_tag>::name, "SHMEM");
}

TEST(BackendTraitsTest, NcclTraitsExist) {
    EXPECT_TRUE(backend_traits<nccl_backend_tag>::supports_collectives);
    EXPECT_TRUE(backend_traits<nccl_backend_tag>::supports_gpu_aware);
    EXPECT_STREQ(backend_traits<nccl_backend_tag>::name, "NCCL");
}

TEST(BackendTraitsTest, SharedMemoryTraitsExist) {
    EXPECT_TRUE(backend_traits<shared_memory_backend_tag>::supports_point_to_point);
    EXPECT_TRUE(backend_traits<shared_memory_backend_tag>::supports_collectives);
    EXPECT_STREQ(backend_traits<shared_memory_backend_tag>::name, "SharedMemory");
}

TEST(BackendTraitsTest, DefaultTraitsAllFalse) {
    // Unspecialized backend falls through to default template
    struct unknown_tag {};
    EXPECT_FALSE(backend_traits<unknown_tag>::supports_point_to_point);
    EXPECT_FALSE(backend_traits<unknown_tag>::supports_collectives);
    EXPECT_FALSE(backend_traits<unknown_tag>::supports_rma);
    EXPECT_STREQ(backend_traits<unknown_tag>::name, "unknown");
}

TEST(BackendTraitsTest, CombinedTraits) {
    using combined = combined_backend_traits<mpi_backend_tag, cuda_backend_tag>;
    // Both must support for && to be true
    EXPECT_FALSE(combined::supports_point_to_point);  // CUDA doesn't
    EXPECT_TRUE(combined::supports_gpu_aware);  // ||  -> CUDA does
    EXPECT_TRUE(combined::supports_async);  // Both do
}

TEST(BackendTraitsTest, HipTraitsSymmetricWithCuda) {
    // HIP should mirror CUDA's capabilities
    EXPECT_EQ(backend_traits<hip_backend_tag>::supports_gpu_aware,
              backend_traits<cuda_backend_tag>::supports_gpu_aware);
    EXPECT_EQ(backend_traits<hip_backend_tag>::supports_async,
              backend_traits<cuda_backend_tag>::supports_async);
    EXPECT_EQ(backend_traits<hip_backend_tag>::supports_thread_multiple,
              backend_traits<cuda_backend_tag>::supports_thread_multiple);
}

// =============================================================================
// Basic Topology Tests (T05)
// =============================================================================

TEST(BasicTopologyTest, SatisfiesTopologyConcept) {
    // Compile-time check is in the header; runtime verification here
    basic_topology topo;
    EXPECT_GE(topo.num_nodes(), 1u);
    EXPECT_GE(topo.num_cores(), 1u);
    EXPECT_GE(topo.num_pus(), 1u);
}

TEST(BasicTopologyTest, NumCoresNotStub) {
    // Should return real hardware concurrency, not hardcoded 1
    auto cores = basic_topology::num_cores();
    auto hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        EXPECT_EQ(cores, static_cast<size_type>(hw));
    } else {
        EXPECT_GE(cores, 1u);
    }
}

TEST(BasicTopologyTest, NumPusNotStub) {
    auto pus = basic_topology::num_pus();
    auto hw = std::thread::hardware_concurrency();
    if (hw > 0) {
        EXPECT_EQ(pus, static_cast<size_type>(hw));
    } else {
        EXPECT_GE(pus, 1u);
    }
}

TEST(BasicTopologyTest, HostnameNotStub) {
    auto name = basic_topology::hostname();
    EXPECT_FALSE(name.empty());
    // On Linux, hostname should not be "localhost" (unless it actually is)
    // We can't assert != "localhost" because CI might set it, but we can
    // check it's non-empty
    EXPECT_GE(name.size(), 1u);
}

TEST(BasicTopologyTest, NumSocketsReasonable) {
    auto sockets = basic_topology::num_sockets();
    EXPECT_GE(sockets, 1u);
    EXPECT_LE(sockets, 256u);  // Reasonable upper bound
}

TEST(BasicTopologyTest, ResultsCached) {
    // Calling multiple times should return same value (cached)
    auto cores1 = basic_topology::num_cores();
    auto cores2 = basic_topology::num_cores();
    EXPECT_EQ(cores1, cores2);

    auto host1 = basic_topology::hostname();
    auto host2 = basic_topology::hostname();
    EXPECT_EQ(host1, host2);
}

TEST(BasicTopologyTest, GpuCountZeroWithoutRuntime) {
    // Without CUDA/HIP runtime, should return 0
    auto gpus = basic_topology::num_gpus();
    EXPECT_EQ(gpus, 0u);
}

// =============================================================================
// Event Concept Tests (T02)
// =============================================================================

TEST(EventConceptTest, NullEventSatisfiesConcept) {
    // null_event is the reference
    static_assert(Event<null_event>, "null_event must satisfy Event concept");

    null_event evt;
    evt.wait();
    EXPECT_EQ(evt.query(), event_status::complete);
    evt.synchronize();
}

// =============================================================================
// Executor Concept Tests (T01)
// =============================================================================

TEST(Phase07ExecutorConceptTest, InlineExecutorSatisfiesExecutorConcept) {
    static_assert(Executor<inline_executor>,
                  "inline_executor must satisfy Executor concept");

    inline_executor exec;
    int val = 0;
    exec.execute([&val]{ val = 42; });
    EXPECT_EQ(val, 42);
    EXPECT_STREQ(exec.name(), "inline");
}

TEST(Phase07ExecutorConceptTest, SequentialExecutorSatisfiesExecutorConcept) {
    static_assert(Executor<sequential_executor>,
                  "sequential_executor must satisfy Executor concept");
    static_assert(ParallelExecutor<sequential_executor>,
                  "sequential_executor must satisfy ParallelExecutor concept");
}

// =============================================================================
// Static Assert Compile-Time Tests (T04)
// =============================================================================
// These are compile-time checks; if the file compiles, they pass.

// Standard executors
static_assert(Executor<inline_executor>,
              "inline_executor must satisfy Executor concept");
static_assert(Executor<sequential_executor>,
              "sequential_executor must satisfy Executor concept");

// Standard events
static_assert(Event<null_event>,
              "null_event must satisfy Event concept");

// Basic topology
static_assert(Topology<basic_topology>,
              "basic_topology must satisfy Topology concept");

// Backend traits existence
static_assert(backend_traits<mpi_backend_tag>::supports_collectives);
static_assert(backend_traits<hip_backend_tag>::supports_gpu_aware);
static_assert(backend_traits<shmem_backend_tag>::supports_rma);

}  // namespace dtl::test
