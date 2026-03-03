// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_backend_coverage.cpp
/// @brief Unit tests for backend concepts, tags, traits, and executor types
/// @details Phase 14 T02: Communicator/Executor/MemorySpace concept satisfaction,
///          backend_traits queries, executor_properties, and standard executors.

#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/backend/concepts/executor.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/common/backend_traits.hpp>

#include <gtest/gtest.h>

#include <concepts>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>

namespace dtl::test {

// =============================================================================
// Mock types for concept satisfaction checks
// =============================================================================

/// @brief Mock communicator that satisfies the Communicator concept
struct mock_communicator {
    using size_type = dtl::size_type;

    rank_t rank() const { return 0; }
    rank_t size() const { return 1; }
    void send(const void*, size_type, rank_t, int) {}
    void recv(void*, size_type, rank_t, int) {}
    request_handle isend(const void*, size_type, rank_t, int) { return {}; }
    request_handle irecv(void*, size_type, rank_t, int) { return {}; }
    void wait(request_handle&) {}
    bool test(request_handle&) { return true; }
};

/// @brief Mock collective communicator
struct mock_collective_communicator : mock_communicator {
    void barrier() {}
    void broadcast(void*, size_type, rank_t) {}
    void scatter(const void*, void*, size_type, rank_t) {}
    void gather(const void*, void*, size_type, rank_t) {}
    void allgather(const void*, void*, size_type) {}
    void alltoall(const void*, void*, size_type) {}
};

/// @brief Incomplete communicator (missing operations)
struct incomplete_communicator {
    using size_type = dtl::size_type;
    rank_t rank() const { return 0; }
    // missing size(), send(), recv(), etc.
};

/// @brief Mock memory space satisfying MemorySpace concept
struct mock_memory_space {
    using pointer = void*;
    using size_type = dtl::size_type;

    void* allocate(size_type sz) { return std::malloc(sz); }
    void* allocate(size_type sz, size_type /*alignment*/) { return std::malloc(sz); }
    void deallocate(void* p, size_type) { std::free(p); }
    memory_space_properties properties() const { return {}; }
    const char* name() const { return "mock"; }
};

/// @brief Mock accessible memory space
struct mock_accessible_memory_space : mock_memory_space {
    bool is_host_accessible() const { return true; }
    bool is_device_accessible() const { return false; }
    bool is_accessible_from_host() const { return true; }
    bool is_accessible_from_device() const { return false; }
};

// =============================================================================
// Communicator Concept Tests
// =============================================================================

TEST(BackendConceptTest, RequestHandleDefaultInvalid) {
    request_handle rh;
    EXPECT_FALSE(rh.valid());
    EXPECT_EQ(rh.handle, nullptr);
}

TEST(BackendConceptTest, RequestHandleValid) {
    int dummy = 42;
    request_handle rh;
    rh.handle = &dummy;
    EXPECT_TRUE(rh.valid());
}

TEST(BackendConceptTest, MockCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_communicator>,
                  "mock_communicator should satisfy Communicator concept");
    SUCCEED();
}

TEST(BackendConceptTest, MockCollectiveCommunicatorSatisfiesConcept) {
    static_assert(CollectiveCommunicator<mock_collective_communicator>,
                  "mock_collective_communicator should satisfy CollectiveCommunicator concept");
    SUCCEED();
}

TEST(BackendConceptTest, IncompleteCommunicatorDoesNotSatisfy) {
    static_assert(!Communicator<incomplete_communicator>,
                  "incomplete_communicator should NOT satisfy Communicator concept");
    SUCCEED();
}

TEST(BackendConceptTest, IntDoesNotSatisfyCommunicator) {
    static_assert(!Communicator<int>,
                  "int should NOT satisfy Communicator concept");
    SUCCEED();
}

TEST(BackendConceptTest, CollectiveCommunicatorRequiresCommunicator) {
    // CollectiveCommunicator refines Communicator, so a plain int doesn't satisfy it
    static_assert(!CollectiveCommunicator<int>,
                  "int should NOT satisfy CollectiveCommunicator concept");
    SUCCEED();
}

// =============================================================================
// Executor Concept Tests
// =============================================================================

TEST(BackendConceptTest, InlineExecutorSatisfiesExecutor) {
    static_assert(Executor<inline_executor>,
                  "inline_executor should satisfy Executor concept");
    SUCCEED();
}

TEST(BackendConceptTest, InlineExecutorSatisfiesSyncExecutor) {
    static_assert(SyncExecutor<inline_executor>,
                  "inline_executor should satisfy SyncExecutor concept");
    SUCCEED();
}

TEST(BackendConceptTest, SequentialExecutorSatisfiesExecutor) {
    static_assert(Executor<sequential_executor>,
                  "sequential_executor should satisfy Executor concept");
    SUCCEED();
}

TEST(BackendConceptTest, SequentialExecutorSatisfiesParallelExecutor) {
    static_assert(ParallelExecutor<sequential_executor>,
                  "sequential_executor should satisfy ParallelExecutor concept");
    SUCCEED();
}

TEST(BackendConceptTest, IntDoesNotSatisfyExecutor) {
    static_assert(!Executor<int>,
                  "int should NOT satisfy Executor concept");
    SUCCEED();
}

// =============================================================================
// Executor Functionality Tests
// =============================================================================

TEST(InlineExecutorTest, ExecuteRunsImmediately) {
    inline_executor exec;
    int counter = 0;
    exec.execute([&] { counter = 42; });
    EXPECT_EQ(counter, 42);
}

TEST(InlineExecutorTest, NameIsInline) {
    EXPECT_STREQ(inline_executor::name(), "inline");
}

TEST(InlineExecutorTest, IsSynchronousTrue) {
    EXPECT_TRUE(inline_executor::is_synchronous());
}

TEST(SequentialExecutorTest, ExecuteWorks) {
    sequential_executor exec;
    int counter = 0;
    exec.execute([&] { counter = 99; });
    EXPECT_EQ(counter, 99);
}

TEST(SequentialExecutorTest, NameIsSequential) {
    EXPECT_STREQ(sequential_executor::name(), "sequential");
}

TEST(SequentialExecutorTest, MaxParallelismIsOne) {
    EXPECT_EQ(sequential_executor::max_parallelism(), 1u);
}

TEST(SequentialExecutorTest, SuggestedParallelismIsOne) {
    EXPECT_EQ(sequential_executor::suggested_parallelism(), 1u);
}

TEST(SequentialExecutorTest, ParallelForIteratesAll) {
    sequential_executor exec;
    int sum = 0;
    exec.parallel_for(static_cast<size_type>(5), [&](size_type i) { sum += static_cast<int>(i); });
    // 0 + 1 + 2 + 3 + 4 = 10
    EXPECT_EQ(sum, 10);
}

TEST(SequentialExecutorTest, ParallelForZeroCount) {
    sequential_executor exec;
    int counter = 0;
    exec.parallel_for(static_cast<size_type>(0), [&](size_type) { counter++; });
    EXPECT_EQ(counter, 0);
}

// =============================================================================
// Executor Properties Tests
// =============================================================================

TEST(ExecutorPropertiesTest, DefaultValues) {
    executor_properties props;
    EXPECT_EQ(props.max_concurrency, 1u);
    EXPECT_TRUE(props.in_order);
    EXPECT_FALSE(props.owns_threads);
    EXPECT_FALSE(props.supports_work_stealing);
}

// =============================================================================
// Executor Traits Tests
// =============================================================================

TEST(ExecutorTraitsTest, DefaultTraitsAreDisabled) {
    // Unspecialized traits should have everything as false
    using traits = executor_traits<int>;
    EXPECT_FALSE(traits::is_sync);
    EXPECT_FALSE(traits::is_parallel);
    EXPECT_FALSE(traits::is_gpu);
    EXPECT_EQ(traits::default_chunk_size, 1u);
}

// =============================================================================
// Executor Tag Types Tests
// =============================================================================

TEST(ExecutorTagTypesTest, TagsAreDistinct) {
    static_assert(!std::is_same_v<inline_executor_tag, thread_pool_executor_tag>);
    static_assert(!std::is_same_v<inline_executor_tag, single_thread_executor_tag>);
    static_assert(!std::is_same_v<inline_executor_tag, gpu_executor_tag>);
    static_assert(!std::is_same_v<thread_pool_executor_tag, gpu_executor_tag>);
    SUCCEED();
}

// =============================================================================
// MemorySpace Concept Tests
// =============================================================================

TEST(BackendConceptTest, MockMemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<mock_memory_space>,
                  "mock_memory_space should satisfy MemorySpace concept");
    SUCCEED();
}

TEST(BackendConceptTest, MockAccessibleMemorySpaceSatisfiesConcept) {
    static_assert(AccessibleMemorySpace<mock_accessible_memory_space>,
                  "mock_accessible_memory_space should satisfy AccessibleMemorySpace");
    SUCCEED();
}

TEST(BackendConceptTest, IntDoesNotSatisfyMemorySpace) {
    static_assert(!MemorySpace<int>,
                  "int should NOT satisfy MemorySpace concept");
    SUCCEED();
}

// =============================================================================
// Memory Space Properties Tests
// =============================================================================

TEST(MemorySpacePropertiesCoverageTest, DefaultProperties) {
    memory_space_properties props;
    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_FALSE(props.unified);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_TRUE(props.pageable);
    EXPECT_EQ(props.alignment, alignof(std::max_align_t));
}

// =============================================================================
// Memory Space Tag Types Tests
// =============================================================================

TEST(MemorySpaceTagTypesTest, TagsAreDistinct) {
    static_assert(!std::is_same_v<host_memory_space_tag, device_memory_space_tag>);
    static_assert(!std::is_same_v<host_memory_space_tag, unified_memory_space_tag>);
    static_assert(!std::is_same_v<host_memory_space_tag, pinned_memory_space_tag>);
    static_assert(!std::is_same_v<device_memory_space_tag, unified_memory_space_tag>);
    SUCCEED();
}

// =============================================================================
// Memory Space Traits Tests
// =============================================================================

TEST(MemorySpaceTraitsTest, DefaultTraitsAreFalse) {
    using traits = memory_space_traits<int>;
    EXPECT_FALSE(traits::is_host_space);
    EXPECT_FALSE(traits::is_device_space);
    EXPECT_FALSE(traits::is_unified_space);
    EXPECT_TRUE(traits::is_thread_safe);
}

// =============================================================================
// Memory Space Utilities Tests
// =============================================================================

TEST(MemorySpaceUtilsTest, SpacesCompatible) {
    bool compat = spaces_compatible<mock_memory_space, mock_memory_space>();
    EXPECT_TRUE(compat);
}

TEST(MemorySpaceUtilsTest, SpaceAlignmentForInt) {
    auto alignment = space_alignment<mock_memory_space, int>();
    EXPECT_EQ(alignment, alignof(int));
}

TEST(MemorySpaceUtilsTest, SpaceAlignmentForDouble) {
    auto alignment = space_alignment<mock_memory_space, double>();
    EXPECT_EQ(alignment, alignof(double));
}

// =============================================================================
// Backend Traits Tests
// =============================================================================

TEST(BackendTraitsTest, UnknownBackendDefaults) {
    using traits = backend_traits<int>;
    EXPECT_FALSE(traits::supports_point_to_point);
    EXPECT_FALSE(traits::supports_collectives);
    EXPECT_FALSE(traits::supports_rma);
    EXPECT_FALSE(traits::supports_gpu_aware);
    EXPECT_FALSE(traits::supports_async);
    EXPECT_FALSE(traits::supports_thread_multiple);
    EXPECT_FALSE(traits::supports_rdma);
    EXPECT_STREQ(traits::name, "unknown");
}

TEST(BackendTraitsTest, MpiBackendTraits) {
    using traits = backend_traits<mpi_backend_tag>;
    EXPECT_TRUE(traits::supports_point_to_point);
    EXPECT_TRUE(traits::supports_collectives);
    EXPECT_TRUE(traits::supports_rma);
    EXPECT_FALSE(traits::supports_gpu_aware);
    EXPECT_TRUE(traits::supports_async);
    EXPECT_TRUE(traits::supports_thread_multiple);
    EXPECT_STREQ(traits::name, "MPI");
}

TEST(BackendTraitsTest, CudaBackendTraits) {
    using traits = backend_traits<cuda_backend_tag>;
    EXPECT_FALSE(traits::supports_point_to_point);
    EXPECT_FALSE(traits::supports_collectives);
    EXPECT_TRUE(traits::supports_gpu_aware);
    EXPECT_TRUE(traits::supports_async);
    EXPECT_STREQ(traits::name, "CUDA");
}

TEST(BackendTraitsTest, NcclBackendTraits) {
    using traits = backend_traits<nccl_backend_tag>;
    EXPECT_TRUE(traits::supports_point_to_point);
    EXPECT_TRUE(traits::supports_collectives);
    EXPECT_TRUE(traits::supports_gpu_aware);
    EXPECT_TRUE(traits::supports_async);
    EXPECT_TRUE(traits::supports_rdma);
    EXPECT_STREQ(traits::name, "NCCL");
}

TEST(BackendTraitsTest, HipBackendTraits) {
    using traits = backend_traits<hip_backend_tag>;
    EXPECT_TRUE(traits::supports_gpu_aware);
    EXPECT_TRUE(traits::supports_async);
    EXPECT_STREQ(traits::name, "HIP");
}

TEST(BackendTraitsTest, ShmemBackendTraits) {
    using traits = backend_traits<shmem_backend_tag>;
    EXPECT_TRUE(traits::supports_rma);
    EXPECT_TRUE(traits::supports_rdma);
    EXPECT_STREQ(traits::name, "SHMEM");
}

TEST(BackendTraitsTest, SharedMemoryBackendTraits) {
    using traits = backend_traits<shared_memory_backend_tag>;
    EXPECT_TRUE(traits::supports_point_to_point);
    EXPECT_TRUE(traits::supports_collectives);
    EXPECT_TRUE(traits::supports_rma);
    EXPECT_TRUE(traits::supports_async);
    EXPECT_TRUE(traits::supports_thread_multiple);
    EXPECT_STREQ(traits::name, "SharedMemory");
}

// =============================================================================
// Backend Variable Templates Tests
// =============================================================================

TEST(BackendTraitsTest, SupportsGpuAwareV) {
    EXPECT_TRUE(supports_gpu_aware_v<cuda_backend_tag>);
    EXPECT_TRUE(supports_gpu_aware_v<hip_backend_tag>);
    EXPECT_FALSE(supports_gpu_aware_v<mpi_backend_tag>);
}

TEST(BackendTraitsTest, SupportsAsyncV) {
    EXPECT_TRUE(supports_async_v<mpi_backend_tag>);
    EXPECT_TRUE(supports_async_v<cuda_backend_tag>);
    EXPECT_FALSE(supports_async_v<shmem_backend_tag>);
}

TEST(BackendTraitsTest, SupportsCollectivesV) {
    EXPECT_TRUE(supports_collectives_v<mpi_backend_tag>);
    EXPECT_TRUE(supports_collectives_v<nccl_backend_tag>);
    EXPECT_FALSE(supports_collectives_v<cuda_backend_tag>);
}

// =============================================================================
// Combined Backend Traits Tests
// =============================================================================

TEST(CombinedBackendTraitsTest, MpiAndCuda) {
    using combined = combined_backend_traits<mpi_backend_tag, cuda_backend_tag>;
    // point_to_point: MPI=true, CUDA=false => combined=false (AND)
    EXPECT_FALSE(combined::supports_point_to_point);
    // gpu_aware: MPI=false, CUDA=true => combined=true (OR)
    EXPECT_TRUE(combined::supports_gpu_aware);
    // async: MPI=true, CUDA=true => combined=true (AND)
    EXPECT_TRUE(combined::supports_async);
}

TEST(CombinedBackendTraitsTest, MpiAndNccl) {
    using combined = combined_backend_traits<mpi_backend_tag, nccl_backend_tag>;
    EXPECT_TRUE(combined::supports_point_to_point);
    EXPECT_TRUE(combined::supports_collectives);
    EXPECT_TRUE(combined::supports_gpu_aware);  // NCCL has it
    EXPECT_TRUE(combined::supports_async);
}

// =============================================================================
// Backend Tag Type Distinctness Tests
// =============================================================================

TEST(BackendTagTypesTest, AllTagsDistinct) {
    static_assert(!std::is_same_v<mpi_backend_tag, cuda_backend_tag>);
    static_assert(!std::is_same_v<mpi_backend_tag, hip_backend_tag>);
    static_assert(!std::is_same_v<mpi_backend_tag, sycl_backend_tag>);
    static_assert(!std::is_same_v<mpi_backend_tag, nccl_backend_tag>);
    static_assert(!std::is_same_v<mpi_backend_tag, shared_memory_backend_tag>);
    static_assert(!std::is_same_v<mpi_backend_tag, shmem_backend_tag>);
    static_assert(!std::is_same_v<cuda_backend_tag, hip_backend_tag>);
    SUCCEED();
}

}  // namespace dtl::test
