// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mock_single_rank_comm.hpp
/// @brief Mock single-rank communicator for unit testing distributed algorithms
/// @details Satisfies the Communicator concept for single-rank testing.
///          All collective operations are identity/no-op.

#pragma once

#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#include <cstring>

namespace dtl::test {

/// @brief Single-rank mock communicator satisfying the Communicator concept
struct mock_single_rank_comm {
    using size_type = dtl::size_type;

    rank_t my_rank_ = 0;
    rank_t num_ranks_ = 1;

    [[nodiscard]] rank_t rank() const noexcept { return my_rank_; }
    [[nodiscard]] rank_t size() const noexcept { return num_ranks_; }

    // Blocking P2P (no-op for single rank)
    void send(const void* /*buf*/, size_type /*count*/,
              rank_t /*dest*/, int /*tag*/) const {}

    void recv(void* /*buf*/, size_type /*count*/,
              rank_t /*source*/, int /*tag*/) const {}

    // Non-blocking P2P
    request_handle isend(const void* /*buf*/, size_type /*count*/,
                         rank_t /*dest*/, int /*tag*/) const {
        return request_handle{};
    }

    request_handle irecv(void* /*buf*/, size_type /*count*/,
                         rank_t /*source*/, int /*tag*/) const {
        return request_handle{};
    }

    // Request completion
    void wait(request_handle& /*req*/) const {}
    bool test(request_handle& /*req*/) const { return true; }

    // Synchronization
    void barrier() const {}

    // Collective operations (single-rank identity)
    void broadcast(void* /*buf*/, size_type /*count*/, rank_t /*root*/) const {}

    void scatter(const void* sendbuf, void* recvbuf,
                 size_type count, rank_t /*root*/) const {
        std::memcpy(recvbuf, sendbuf, count);
    }

    void gather(const void* sendbuf, void* recvbuf,
                size_type count, rank_t /*root*/) const {
        std::memcpy(recvbuf, sendbuf, count);
    }

    void allgather(const void* sendbuf, void* recvbuf,
                   size_type elem_size) const {
        std::memcpy(recvbuf, sendbuf, elem_size);
    }

    void allgatherv(const void* sendbuf, size_type sendcount,
                    void* recvbuf, const int* /*recv_counts*/,
                    const int* /*recv_displs*/,
                    size_type elem_size) const {
        std::memcpy(recvbuf, sendbuf, sendcount * elem_size);
    }

    void gatherv(const void* sendbuf, size_type sendcount,
                 void* recvbuf, const int* /*recv_counts*/,
                 const int* /*recv_displs*/,
                 size_type elem_size, rank_t /*root*/) const {
        std::memcpy(recvbuf, sendbuf, sendcount * elem_size);
    }

    void alltoall(const void* sendbuf, void* recvbuf,
                  size_type elem_size) const {
        std::memcpy(recvbuf, sendbuf, elem_size);
    }

    void alltoallv(const void* sendbuf, const int* /*send_counts*/,
                   const int* /*send_displs*/, void* recvbuf,
                   const int* recv_counts, const int* /*recv_displs*/,
                   size_type elem_size) const {
        // Single rank: total recv = recv_counts[0]
        std::memcpy(recvbuf, sendbuf,
                    static_cast<size_type>(recv_counts[0]) * elem_size);
    }

    // Reduction operations
    void reduce_sum(const void* sendbuf, void* recvbuf,
                    size_type count, rank_t /*root*/) const {
        std::memcpy(recvbuf, sendbuf, count);
    }

    void allreduce_sum(const void* sendbuf, void* recvbuf,
                       size_type count) const {
        std::memcpy(recvbuf, sendbuf, count);
    }

    template <typename T>
    T allreduce_sum_value(const T& val) const { return val; }

    bool allreduce_land_value(bool val) const { return val; }
    bool allreduce_lor_value(bool val) const { return val; }
};

// Verify the mock satisfies the Communicator concept
static_assert(Communicator<mock_single_rank_comm>,
              "mock_single_rank_comm must satisfy Communicator concept");

}  // namespace dtl::test
