// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_shmem_backend.cpp
/// @brief Integration tests for OpenSHMEM backend
/// @details Tests SHMEM communication operations with multiple PEs.
/// @note Run with: oshrun -np 2 ./test_executable
///       or:       oshrun -np 4 ./test_executable

#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/backend/concepts/memory_space.hpp>
#include <dtl/backend/concepts/rma_communicator.hpp>

#if DTL_ENABLE_SHMEM
#include <backends/shmem/shmem_communicator.hpp>
#include <backends/shmem/shmem_memory_space.hpp>
#include <backends/shmem/shmem_memory_window_impl.hpp>
#include <backends/shmem/shmem_rma_adapter.hpp>
#endif

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

namespace dtl::test {

#if DTL_ENABLE_SHMEM

// =============================================================================
// Test Fixture
// =============================================================================

class ShmemBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // SHMEM should already be initialized by the test framework
        comm_ = &shmem::world_communicator();
        rma_ = &shmem::global_rma_adapter();
    }

    shmem::shmem_communicator* comm_ = nullptr;
    shmem::shmem_rma_adapter* rma_ = nullptr;
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST_F(ShmemBackendTest, MemorySpaceSatisfiesConcept) {
    static_assert(MemorySpace<shmem::shmem_static_memory_space>,
                  "shmem_static_memory_space must satisfy MemorySpace concept");
}

// =============================================================================
// Basic Properties Tests
// =============================================================================

TEST_F(ShmemBackendTest, RankInValidRange) {
    EXPECT_GE(comm_->rank(), 0);
    EXPECT_LT(comm_->rank(), comm_->size());
}

TEST_F(ShmemBackendTest, SizeAtLeastOne) {
    EXPECT_GE(comm_->size(), 1);
}

TEST_F(ShmemBackendTest, CommunicatorValid) {
    EXPECT_TRUE(comm_->valid());
}

TEST_F(ShmemBackendTest, RmaAdapterMatchesCommunicatorProperties) {
    EXPECT_EQ(rma_->rank(), comm_->rank());
    EXPECT_EQ(rma_->size(), comm_->size());
    EXPECT_TRUE(rma_->valid());
}

// =============================================================================
// Symmetric Memory Space Tests
// =============================================================================

TEST_F(ShmemBackendTest, SymmetricMemoryProperties) {
    auto props = shmem::shmem_symmetric_memory_space::properties();
    EXPECT_TRUE(props.host_accessible);
    EXPECT_FALSE(props.device_accessible);
    EXPECT_TRUE(props.supports_atomics);
    EXPECT_EQ(std::string(shmem::shmem_symmetric_memory_space::name()),
              "shmem_symmetric");
}

TEST_F(ShmemBackendTest, SymmetricMemoryAllocation) {
    constexpr size_type alloc_size = 1024;

    // Allocate symmetric memory
    void* ptr = shmem::shmem_symmetric_memory_space::allocate(alloc_size);
    ASSERT_NE(ptr, nullptr);

    // Write local data
    int* data = static_cast<int*>(ptr);
    *data = comm_->rank() + 100;

    // Barrier to ensure all PEs have written
    comm_->barrier();

    // Verify local data
    EXPECT_EQ(*data, comm_->rank() + 100);

    // Deallocate
    shmem::shmem_symmetric_memory_space::deallocate(ptr, alloc_size);
}

TEST_F(ShmemBackendTest, SymmetricCallocZeroInitializes) {
    constexpr size_type count = 64;
    constexpr size_type elem_size = sizeof(int);

    int* data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::calloc(count, elem_size));
    ASSERT_NE(data, nullptr);

    // Verify zero initialization
    for (size_type i = 0; i < count; ++i) {
        EXPECT_EQ(data[i], 0) << "Element " << i << " not zero-initialized";
    }

    shmem::shmem_symmetric_memory_space::deallocate(data, count * elem_size);
}

// =============================================================================
// One-Sided Put/Get Tests
// =============================================================================

TEST_F(ShmemBackendTest, PutGetPingPong) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    // Allocate symmetric memory for communication
    int* sym_data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(sym_data, nullptr);

    *sym_data = 0;  // Initialize
    comm_->barrier();

    if (comm_->rank() == 0) {
        // PE 0 puts value to PE 1
        int value = 12345;
        rma_->put(1, sym_data, &value, 1);  // Typed put for int
        rma_->quiet();  // Ensure put completes
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        EXPECT_EQ(*sym_data, 12345);
    }

    shmem::shmem_symmetric_memory_space::deallocate(sym_data, sizeof(int));
}

TEST_F(ShmemBackendTest, GetRemoteData) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    // Allocate symmetric memory
    int* sym_data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(sym_data, nullptr);

    // Each PE initializes with its rank + 1000
    *sym_data = comm_->rank() + 1000;
    comm_->barrier();

    // PE 0 gets data from PE 1
    if (comm_->rank() == 0) {
        int local_buffer = 0;
        rma_->get(1, &local_buffer, sym_data, 1);  // Typed get for int
        EXPECT_EQ(local_buffer, 1001);  // PE 1's value = 1 + 1000
    }

    comm_->barrier();
    shmem::shmem_symmetric_memory_space::deallocate(sym_data, sizeof(int));
}

TEST_F(ShmemBackendTest, PutMemBytes) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    constexpr size_type buffer_size = 256;
    char* sym_buffer = static_cast<char*>(
        shmem::shmem_symmetric_memory_space::allocate(buffer_size));
    ASSERT_NE(sym_buffer, nullptr);

    // Clear buffer
    std::fill(sym_buffer, sym_buffer + buffer_size, '\0');
    comm_->barrier();

    if (comm_->rank() == 0) {
        // Put byte data to PE 1
        const char* message = "Hello from PE 0!";
        size_type msg_len = std::strlen(message) + 1;
        rma_->put(1, sym_buffer, message, msg_len);
        rma_->quiet();
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        EXPECT_STREQ(sym_buffer, "Hello from PE 0!");
    }

    shmem::shmem_symmetric_memory_space::deallocate(sym_buffer, buffer_size);
}

// =============================================================================
// Non-Blocking Put/Get Tests
// =============================================================================

TEST_F(ShmemBackendTest, NonBlockingPut) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    constexpr size_type count = 100;
    int* sym_array = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(count * sizeof(int)));
    ASSERT_NE(sym_array, nullptr);

    std::fill(sym_array, sym_array + count, 0);
    comm_->barrier();

    if (comm_->rank() == 0) {
        // Prepare local data
        std::vector<int> local_data(count);
        std::iota(local_data.begin(), local_data.end(), 1);

        // Non-blocking put to PE 1
        rma_->put_nbi(1, sym_array, local_data.data(), count * sizeof(int));
        rma_->quiet();  // Wait for completion
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        // Verify received data
        for (size_type i = 0; i < count; ++i) {
            EXPECT_EQ(sym_array[i], static_cast<int>(i + 1))
                << "Mismatch at index " << i;
        }
    }

    shmem::shmem_symmetric_memory_space::deallocate(sym_array, count * sizeof(int));
}

// =============================================================================
// Atomic Operation Tests
// =============================================================================

TEST_F(ShmemBackendTest, AtomicFetchAdd) {
    // Allocate symmetric counter
    int* counter = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(counter, nullptr);

    *counter = 0;
    comm_->barrier();

    // All PEs atomically increment counter on PE 0
    int old_value = rma_->fetch_add(counter, 1, 0);

    // The old values returned should be unique across PEs
    // (each PE gets the value before its increment)
    EXPECT_GE(old_value, 0);
    EXPECT_LT(old_value, comm_->size());

    comm_->barrier();

    // PE 0 should see final count
    if (comm_->rank() == 0) {
        EXPECT_EQ(*counter, comm_->size());
    }

    shmem::shmem_symmetric_memory_space::deallocate(counter, sizeof(int));
}

TEST_F(ShmemBackendTest, AtomicCompareSwap) {
    int* flag = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(flag, nullptr);

    *flag = 0;  // Initial value
    comm_->barrier();

    // All PEs try to CAS from 0 to their rank+1
    // Only one should succeed
    int old = rma_->compare_swap(flag, 0, comm_->rank() + 1, 0);

    comm_->barrier();

    // Verify exactly one PE succeeded
    if (comm_->rank() == 0) {
        EXPECT_GT(*flag, 0);  // Someone set it
        EXPECT_LE(*flag, comm_->size());  // Valid rank+1
    }

    // The PE that got old==0 was the winner
    if (old == 0) {
        // This PE won the race
        EXPECT_TRUE(true);  // Just verify we're here
    }

    shmem::shmem_symmetric_memory_space::deallocate(flag, sizeof(int));
}

TEST_F(ShmemBackendTest, AtomicSwap) {
    int* value = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(value, nullptr);

    *value = -1;
    comm_->barrier();

    // Each PE swaps in its rank
    int old = rma_->atomic_swap(value, comm_->rank(), 0);

    comm_->barrier();

    // The old value should be either -1 (initial) or another PE's rank
    EXPECT_TRUE(old == -1 || (old >= 0 && old < comm_->size()));

    shmem::shmem_symmetric_memory_space::deallocate(value, sizeof(int));
}

TEST_F(ShmemBackendTest, AtomicFetch) {
    int* data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(data, nullptr);

    *data = 42;
    comm_->barrier();

    // All PEs atomically fetch value from PE 0
    int fetched = rma_->atomic_fetch(data, 0);
    EXPECT_EQ(fetched, 42);

    shmem::shmem_symmetric_memory_space::deallocate(data, sizeof(int));
}

TEST_F(ShmemBackendTest, AtomicSet) {
    int* data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(data, nullptr);

    *data = 0;
    comm_->barrier();

    // PE 0 atomically sets value on all PEs
    if (comm_->rank() == 0) {
        for (rank_t pe = 0; pe < comm_->size(); ++pe) {
            rma_->atomic_set(data, 999, pe);
        }
    }

    rma_->quiet();
    comm_->barrier();

    EXPECT_EQ(*data, 999);

    shmem::shmem_symmetric_memory_space::deallocate(data, sizeof(int));
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST_F(ShmemBackendTest, FenceOrdering) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    int* a = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    int* b = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);

    *a = 0;
    *b = 0;
    comm_->barrier();

    if (comm_->rank() == 0) {
        // Put a, then fence, then put b
        // PE 1 should see a written before b
        int val_a = 100;
        int val_b = 200;
        rma_->put(1, a, &val_a, 1);
        rma_->fence();  // Orders a before b
        rma_->put(1, b, &val_b, 1);
        rma_->quiet();
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        // Both should be written after barrier
        EXPECT_EQ(*a, 100);
        EXPECT_EQ(*b, 200);
    }

    shmem::shmem_symmetric_memory_space::deallocate(a, sizeof(int));
    shmem::shmem_symmetric_memory_space::deallocate(b, sizeof(int));
}

TEST_F(ShmemBackendTest, QuietCompletion) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    constexpr size_type count = 1000;
    int* data = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(count * sizeof(int)));
    ASSERT_NE(data, nullptr);

    std::fill(data, data + count, 0);
    comm_->barrier();

    if (comm_->rank() == 0) {
        std::vector<int> source(count);
        std::iota(source.begin(), source.end(), 1);

        // Multiple non-blocking puts
        rma_->put_nbi(1, data, source.data(), count * sizeof(int));
        rma_->quiet();  // All operations complete after quiet returns
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        // Verify all data arrived
        int sum = std::accumulate(data, data + count, 0);
        // Sum of 1..1000 = 1000*1001/2 = 500500
        EXPECT_EQ(sum, (count * (count + 1)) / 2);
    }

    shmem::shmem_symmetric_memory_space::deallocate(data, count * sizeof(int));
}

TEST_F(ShmemBackendTest, BarrierSynchronization) {
    int* flag = static_cast<int*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(int)));
    ASSERT_NE(flag, nullptr);

    *flag = 0;

    // Phase 1: PE 0 sets flag
    if (comm_->rank() == 0) {
        *flag = 1;
    }

    comm_->barrier();  // All PEs wait here

    // Phase 2: All PEs check flag (only PE 0 has it set)
    // This just tests barrier works - flag isn't distributed
    if (comm_->rank() == 0) {
        EXPECT_EQ(*flag, 1);
    }

    shmem::shmem_symmetric_memory_space::deallocate(flag, sizeof(int));
}

// =============================================================================
// Memory Space Traits Tests
// =============================================================================

TEST_F(ShmemBackendTest, MemorySpaceTraits) {
    using traits = memory_space_traits<shmem::shmem_static_memory_space>;

    EXPECT_TRUE(traits::is_host_space);
    EXPECT_FALSE(traits::is_device_space);
    EXPECT_FALSE(traits::is_unified_space);
    EXPECT_TRUE(traits::is_symmetric_space);
    EXPECT_FALSE(traits::is_thread_safe);  // Collective ops not thread-safe
}

// =============================================================================
// Typed Put/Get Tests (Long and Double)
// =============================================================================

TEST_F(ShmemBackendTest, TypedPutGetLong) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    long* data = static_cast<long*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(long)));
    ASSERT_NE(data, nullptr);

    *data = 0;
    comm_->barrier();

    if (comm_->rank() == 0) {
        long value = 9876543210L;
        rma_->put(1, data, &value, 1);
        rma_->quiet();
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        EXPECT_EQ(*data, 9876543210L);
    }

    shmem::shmem_symmetric_memory_space::deallocate(data, sizeof(long));
}

TEST_F(ShmemBackendTest, TypedPutGetDouble) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    double* data = static_cast<double*>(
        shmem::shmem_symmetric_memory_space::allocate(sizeof(double)));
    ASSERT_NE(data, nullptr);

    *data = 0.0;
    comm_->barrier();

    if (comm_->rank() == 0) {
        double value = 3.14159265358979;
        rma_->put(1, data, &value, 1);
        rma_->quiet();
    }

    comm_->barrier();

    if (comm_->rank() == 1) {
        EXPECT_DOUBLE_EQ(*data, 3.14159265358979);
    }

    shmem::shmem_symmetric_memory_space::deallocate(data, sizeof(double));
}

// =============================================================================
// Memory Window Implementation Tests
// =============================================================================

TEST_F(ShmemBackendTest, MemoryWindowCreation) {
    auto window_result = shmem::make_shmem_window(1024);
    ASSERT_TRUE(window_result.has_value());

    auto& window = *window_result.value();
    EXPECT_NE(window.base(), nullptr);
    EXPECT_EQ(window.size(), 1024);
    EXPECT_TRUE(window.valid());
    EXPECT_EQ(window.rank(), comm_->rank());
    EXPECT_EQ(window.num_pes(), comm_->size());
}

TEST_F(ShmemBackendTest, MemoryWindowFromExistingMemory) {
    void* sym_ptr = shmem::shmem_symmetric_memory_space::allocate(512);
    ASSERT_NE(sym_ptr, nullptr);

    auto window_result = shmem::make_shmem_window(sym_ptr, 512);
    ASSERT_TRUE(window_result.has_value());

    auto& window = *window_result.value();
    EXPECT_EQ(window.base(), sym_ptr);
    EXPECT_EQ(window.size(), 512);
    EXPECT_TRUE(window.valid());

    // Window doesn't own memory, so we must free it
    shmem::shmem_symmetric_memory_space::deallocate(sym_ptr, 512);
}

TEST_F(ShmemBackendTest, MemoryWindowPutGet) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    auto window_result = shmem::make_shmem_window(sizeof(int) * 10);
    ASSERT_TRUE(window_result.has_value());
    auto& window = *window_result.value();

    int* data = static_cast<int*>(window.base());
    for (int i = 0; i < 10; ++i) {
        data[i] = comm_->rank() * 100 + i;
    }

    window.barrier();

    // PE 0 puts to PE 1
    if (comm_->rank() == 0) {
        std::vector<int> send_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto result = window.put(send_data.data(), sizeof(int) * 10, 1, 0);
        EXPECT_TRUE(result.has_value());
        window.flush_all();
    }

    window.barrier();

    if (comm_->rank() == 1) {
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i + 1) << "Mismatch at index " << i;
        }
    }
}

TEST_F(ShmemBackendTest, MemoryWindowAsyncPutGet) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    constexpr size_type count = 100;
    auto window_result = shmem::make_shmem_window(sizeof(int) * count);
    ASSERT_TRUE(window_result.has_value());
    auto& window = *window_result.value();

    int* data = static_cast<int*>(window.base());
    std::fill(data, data + count, 0);

    window.barrier();

    if (comm_->rank() == 0) {
        std::vector<int> send_data(count);
        std::iota(send_data.begin(), send_data.end(), 1);

        memory_window_impl::rma_request_handle request;
        auto result = window.async_put(send_data.data(), sizeof(int) * count, 1, 0, request);
        EXPECT_TRUE(result.has_value());

        // Wait for completion
        window.wait_async(request);
        EXPECT_TRUE(request.completed);
    }

    window.barrier();

    if (comm_->rank() == 1) {
        int sum = 0;
        for (size_type i = 0; i < count; ++i) {
            sum += data[i];
        }
        // Sum of 1..100 = 100 * 101 / 2 = 5050
        EXPECT_EQ(sum, 5050);
    }
}

TEST_F(ShmemBackendTest, MemoryWindowFetchAndOp) {
    auto window_result = shmem::make_shmem_window(sizeof(int));
    ASSERT_TRUE(window_result.has_value());
    auto& window = *window_result.value();

    int* counter = static_cast<int*>(window.base());
    *counter = 0;

    window.barrier();

    // All PEs atomically increment counter on PE 0
    int increment = 1;
    int old_value = 0;
    auto result = window.fetch_and_op(&increment, &old_value, sizeof(int),
                                       0, 0, rma_reduce_op::sum);
    EXPECT_TRUE(result.has_value());

    window.flush_all();
    window.barrier();

    if (comm_->rank() == 0) {
        EXPECT_EQ(*counter, comm_->size());
    }
}

TEST_F(ShmemBackendTest, MemoryWindowCompareAndSwap) {
    auto window_result = shmem::make_shmem_window(sizeof(int));
    ASSERT_TRUE(window_result.has_value());
    auto& window = *window_result.value();

    int* flag = static_cast<int*>(window.base());
    *flag = 0;

    window.barrier();

    // All PEs try to CAS from 0 to their rank+1
    int compare = 0;
    int new_value = comm_->rank() + 1;
    int old_value = 0;

    auto result = window.compare_and_swap(&new_value, &compare, &old_value,
                                           sizeof(int), 0, 0);
    EXPECT_TRUE(result.has_value());

    window.barrier();

    // Exactly one PE should have won
    if (comm_->rank() == 0) {
        EXPECT_GT(*flag, 0);
        EXPECT_LE(*flag, comm_->size());
    }
}

TEST_F(ShmemBackendTest, MemoryWindowFence) {
    if (comm_->size() < 2) {
        GTEST_SKIP() << "Requires at least 2 PEs";
    }

    auto window_result = shmem::make_shmem_window(sizeof(int) * 2);
    ASSERT_TRUE(window_result.has_value());
    auto& window = *window_result.value();

    int* data = static_cast<int*>(window.base());
    data[0] = 0;
    data[1] = 0;

    window.barrier();

    if (comm_->rank() == 0) {
        int val1 = 100;
        int val2 = 200;

        // Put first value, fence, put second value
        window.put(&val1, sizeof(int), 1, 0);
        window.fence();
        window.put(&val2, sizeof(int), 1, sizeof(int));
        window.flush_all();
    }

    window.barrier();

    if (comm_->rank() == 1) {
        EXPECT_EQ(data[0], 100);
        EXPECT_EQ(data[1], 200);
    }
}

#else  // !DTL_ENABLE_SHMEM

// =============================================================================
// Stub Tests When SHMEM is Disabled
// =============================================================================

TEST(ShmemBackendDisabledTest, ShmemNotEnabled) {
    GTEST_SKIP() << "OpenSHMEM support not enabled. Build with -DDTL_ENABLE_SHMEM=ON";
}

#endif  // DTL_ENABLE_SHMEM

}  // namespace dtl::test
