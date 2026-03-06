// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file nccl_comm_adapter.hpp
/// @brief Concept-compliant NCCL communicator adapter
/// @details Wraps nccl_communicator with void-returning methods that throw on error.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <backends/nccl/nccl_communicator.hpp>

#include <cstring>
#include <stdexcept>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtl {
namespace nccl {

// ============================================================================
// Communication Error Exception
// ============================================================================

/// @brief Exception thrown when NCCL communication fails
class communication_error : public std::runtime_error {
public:
    explicit communication_error(const std::string& msg)
        : std::runtime_error(msg) {}

    explicit communication_error(const status& s)
        : std::runtime_error(s.to_string()) {}
};

// ============================================================================
// NCCL Communicator Adapter
// ============================================================================

/// @brief Concept-compliant adapter for explicit NCCL device-buffer collectives
/// @details Provides void-returning methods that throw communication_error on failure.
///          This adapter is intentionally limited to raw device-buffer communication
///          and the base communicator concepts. MPI-style scalar helpers and
///          variable-size collective helpers are deliberately omitted until a
///          device-safe distributed algorithm path exists.
///          Satisfies Communicator, CollectiveCommunicator, and ReducingCommunicator concepts.
class nccl_comm_adapter {
public:
    using size_type = dtl::size_type;

    /// @brief Construct adapter wrapping specific communicator
    explicit nccl_comm_adapter(nccl_communicator& comm)
        : impl_(&comm) {}

    /// @brief Construct adapter owning a communicator shared_ptr
    explicit nccl_comm_adapter(std::shared_ptr<nccl_communicator> comm)
        : impl_(comm.get())
        , owned_impl_(std::move(comm)) {}

    // ------------------------------------------------------------------------
    // Query Operations (Communicator concept)
    // ------------------------------------------------------------------------

    [[nodiscard]] rank_t rank() const noexcept {
        return impl_->rank();
    }

    [[nodiscard]] rank_t size() const noexcept {
        return impl_->size();
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Communication (Communicator concept)
    // ------------------------------------------------------------------------

    void send(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->send_impl(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("NCCL send failed to rank " + std::to_string(dest));
        }
    }

    void recv(void* buf, size_type count, rank_t source, int tag) {
        auto result = impl_->recv_impl(buf, count, 1, source, tag);
        if (!result) {
            throw communication_error("NCCL recv failed from rank " + std::to_string(source));
        }
    }

    [[nodiscard]] request_handle isend(const void* buf, size_type count, rank_t dest, int tag) {
        auto result = impl_->isend(buf, count, 1, dest, tag);
        if (!result) {
            throw communication_error("NCCL isend failed to rank " + std::to_string(dest));
        }
        return *result;
    }

    [[nodiscard]] request_handle irecv(void* buf, size_type count, rank_t source, int tag) {
        auto result = impl_->irecv(buf, count, 1, source, tag);
        if (!result) {
            throw communication_error("NCCL irecv failed from rank " + std::to_string(source));
        }
        return *result;
    }

    void wait(request_handle& req) {
        auto result = impl_->wait_event(req);
        if (!result) {
            throw communication_error("NCCL wait failed");
        }
    }

    [[nodiscard]] bool test(request_handle& req) {
        auto result = impl_->test_event(req);
        if (!result) {
            throw communication_error("NCCL test failed");
        }
        return *result;
    }

    // ------------------------------------------------------------------------
    // Collective Operations (CollectiveCommunicator concept)
    // ------------------------------------------------------------------------

    void barrier() {
        auto result = impl_->barrier();
        if (!result) {
            throw communication_error("NCCL barrier failed");
        }
    }

    void broadcast(void* buf, size_type count, rank_t root) {
        auto result = impl_->broadcast_impl(buf, count, 1, root);
        if (!result) {
            throw communication_error("NCCL broadcast failed");
        }
    }

    void scatter(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->scatter_impl(sendbuf, count, recvbuf, count, 1, root);
        if (!result) {
            throw communication_error("NCCL scatter failed");
        }
    }

    void gather(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->gather_impl(sendbuf, count, recvbuf, count, 1, root);
        if (!result) {
            throw communication_error("NCCL gather failed");
        }
    }

    void allgather(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allgather_impl(sendbuf, count, recvbuf, count, 1);
        if (!result) {
            throw communication_error("NCCL allgather failed");
        }
    }

    void alltoall(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->alltoall_impl(sendbuf, recvbuf, count, 1);
        if (!result) {
            throw communication_error("NCCL alltoall failed");
        }
    }

    // ------------------------------------------------------------------------
    // Reduction Operations (ReducingCommunicator concept)
    // ------------------------------------------------------------------------

    void reduce_sum(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->reduce_sum_impl(sendbuf, recvbuf, count, root);
        if (!result) {
            throw communication_error("NCCL reduce_sum failed");
        }
    }

    void allreduce_sum(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_sum failed");
        }
    }

    // ------------------------------------------------------------------------
    // Utilities
    // ------------------------------------------------------------------------

    [[nodiscard]] bool is_root() const noexcept {
        return rank() == 0;
    }

    [[nodiscard]] nccl_communicator& underlying() noexcept {
        return *impl_;
    }

    [[nodiscard]] const nccl_communicator& underlying() const noexcept {
        return *impl_;
    }

private:
    nccl_communicator* impl_;
    std::shared_ptr<nccl_communicator> owned_impl_;
};

// ============================================================================
// Concept Verification
// ============================================================================

static_assert(Communicator<nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy Communicator concept");
static_assert(CollectiveCommunicator<nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy CollectiveCommunicator concept");
static_assert(ReducingCommunicator<nccl_comm_adapter>,
              "nccl_comm_adapter must satisfy ReducingCommunicator concept");

}  // namespace nccl
}  // namespace dtl
