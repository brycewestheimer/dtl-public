// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_communicator_concept.cpp
/// @brief Unit tests for Communicator concept
/// @details Verifies concept requirements using mock implementations.

#include <dtl/backend/concepts/communicator.hpp>

#include <gtest/gtest.h>

namespace dtl::test {

// =============================================================================
// Mock Communicator Implementation
// =============================================================================

/// @brief Minimal mock communicator that satisfies the Communicator concept
struct mock_communicator {
    using size_type = dtl::size_type;

    // Query operations
    [[nodiscard]] rank_t rank() const { return rank_; }
    [[nodiscard]] rank_t size() const { return size_; }

    // Blocking point-to-point
    void send(const void* /*buf*/, size_type /*count*/, rank_t /*dest*/, int /*tag*/) {
        // Mock: do nothing
    }

    void recv(void* /*buf*/, size_type /*count*/, rank_t /*src*/, int /*tag*/) {
        // Mock: do nothing
    }

    // Non-blocking point-to-point
    [[nodiscard]] request_handle isend(const void* /*buf*/, size_type /*count*/,
                                        rank_t /*dest*/, int /*tag*/) {
        return request_handle{reinterpret_cast<void*>(1)};
    }

    [[nodiscard]] request_handle irecv(void* /*buf*/, size_type /*count*/,
                                        rank_t /*src*/, int /*tag*/) {
        return request_handle{reinterpret_cast<void*>(2)};
    }

    // Request completion
    void wait(request_handle& req) {
        req.handle = nullptr;  // Mark as completed
    }

    [[nodiscard]] bool test(request_handle& req) {
        if (req.handle != nullptr) {
            req.handle = nullptr;
            return true;
        }
        return false;
    }

private:
    rank_t rank_ = 0;
    rank_t size_ = 1;
};

/// @brief Mock communicator with collective operations
struct mock_collective_communicator : mock_communicator {
    // Synchronization
    void barrier() {
        // Mock barrier
    }

    // Data movement
    void broadcast(void* /*buf*/, size_type /*count*/, rank_t /*root*/) {}
    void scatter(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/, rank_t /*root*/) {}
    void gather(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/, rank_t /*root*/) {}
    void allgather(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/) {}
    void alltoall(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/) {}
};

/// @brief Mock communicator with reduction operations
struct mock_reducing_communicator : mock_collective_communicator {
    void reduce_sum(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/, rank_t /*root*/) {}
    void allreduce_sum(const void* /*sendbuf*/, void* /*recvbuf*/, size_type /*count*/) {}
};

/// @brief Mock communicator with async collectives
struct mock_async_communicator : mock_reducing_communicator {
    [[nodiscard]] request_handle ibarrier() {
        return request_handle{reinterpret_cast<void*>(10)};
    }

    [[nodiscard]] request_handle ibroadcast(void* /*buf*/, size_type /*count*/, rank_t /*root*/) {
        return request_handle{reinterpret_cast<void*>(11)};
    }

    [[nodiscard]] request_handle iscatter(const void* /*sendbuf*/, void* /*recvbuf*/,
                                           size_type /*count*/, rank_t /*root*/) {
        return request_handle{reinterpret_cast<void*>(12)};
    }

    [[nodiscard]] request_handle igather(const void* /*sendbuf*/, void* /*recvbuf*/,
                                          size_type /*count*/, rank_t /*root*/) {
        return request_handle{reinterpret_cast<void*>(13)};
    }
};

/// @brief Type that doesn't satisfy Communicator concept (missing recv)
struct not_a_communicator {
    using size_type = dtl::size_type;
    [[nodiscard]] rank_t rank() const { return 0; }
    [[nodiscard]] rank_t size() const { return 1; }
    void send(const void*, size_type, rank_t, int) {}
    // Missing: recv, isend, irecv, wait, test
};

// =============================================================================
// Concept Verification Tests
// =============================================================================

TEST(CommunicatorConceptTest, MockCommunicatorSatisfiesConcept) {
    static_assert(Communicator<mock_communicator>);
}

TEST(CommunicatorConceptTest, MockCollectiveSatisfiesConcept) {
    static_assert(Communicator<mock_collective_communicator>);
    static_assert(CollectiveCommunicator<mock_collective_communicator>);
}

TEST(CommunicatorConceptTest, MockReducingSatisfiesConcept) {
    static_assert(Communicator<mock_reducing_communicator>);
    static_assert(CollectiveCommunicator<mock_reducing_communicator>);
    static_assert(ReducingCommunicator<mock_reducing_communicator>);
}

TEST(CommunicatorConceptTest, MockAsyncSatisfiesConcept) {
    static_assert(Communicator<mock_async_communicator>);
    static_assert(CollectiveCommunicator<mock_async_communicator>);
    static_assert(AsyncCommunicator<mock_async_communicator>);
}

TEST(CommunicatorConceptTest, NonCommunicatorDoesNotSatisfy) {
    static_assert(!Communicator<not_a_communicator>);
    static_assert(!Communicator<int>);
    static_assert(!Communicator<void>);
}

TEST(CommunicatorConceptTest, BasicCommunicatorDoesNotSatisfyCollective) {
    static_assert(!CollectiveCommunicator<mock_communicator>);
}

// =============================================================================
// Request Handle Tests
// =============================================================================

TEST(RequestHandleTest, DefaultConstruction) {
    request_handle req;
    EXPECT_FALSE(req.valid());
    EXPECT_EQ(req.handle, nullptr);
}

TEST(RequestHandleTest, ValidCheck) {
    request_handle req{reinterpret_cast<void*>(1)};
    EXPECT_TRUE(req.valid());
}

// =============================================================================
// Mock Operation Tests
// =============================================================================

TEST(CommunicatorConceptTest, BasicOperations) {
    mock_communicator comm;

    EXPECT_EQ(comm.rank(), 0);
    EXPECT_EQ(comm.size(), 1);

    int buf = 42;
    comm.send(&buf, sizeof(buf), 0, 0);
    comm.recv(&buf, sizeof(buf), 0, 0);

    auto req = comm.isend(&buf, sizeof(buf), 0, 0);
    EXPECT_TRUE(req.valid());

    comm.wait(req);
    EXPECT_FALSE(req.valid());
}

TEST(CommunicatorConceptTest, TestRequest) {
    mock_communicator comm;

    int buf = 42;
    auto req = comm.irecv(&buf, sizeof(buf), 0, 0);
    EXPECT_TRUE(req.valid());

    bool completed = comm.test(req);
    EXPECT_TRUE(completed);
    EXPECT_FALSE(req.valid());

    // Testing again should return false
    completed = comm.test(req);
    EXPECT_FALSE(completed);
}

// =============================================================================
// Tag Type Tests
// =============================================================================

TEST(CommunicatorTagTest, TagsAreDistinct) {
    static_assert(!std::is_same_v<mpi_communicator_tag, shared_memory_communicator_tag>);
    static_assert(!std::is_same_v<mpi_communicator_tag, gpu_communicator_tag>);
    static_assert(!std::is_same_v<shared_memory_communicator_tag, gpu_communicator_tag>);
}

}  // namespace dtl::test
