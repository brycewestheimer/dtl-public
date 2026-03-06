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

/// @brief Concept-compliant adapter for nccl_communicator
/// @details Provides void-returning methods that throw communication_error on failure.
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
        impl_->wait_event(req);
    }

    [[nodiscard]] bool test(request_handle& req) {
        return impl_->test_event(req);
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
    // Extended Reduction Operations (typed int/long variants)
    // ------------------------------------------------------------------------

    void allreduce_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_sum_int failed");
        }
    }

    void allreduce_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_sum_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_sum_long failed");
        }
    }

    void reduce_sum_int(const void* sendbuf, void* recvbuf, size_type count, rank_t root) {
        auto result = impl_->reduce_sum_int_impl(sendbuf, recvbuf, count, root);
        if (!result) {
            throw communication_error("NCCL reduce_sum_int failed");
        }
    }

    // Min reductions
    void allreduce_min(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_min failed");
        }
    }

    void allreduce_min_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_min_int failed");
        }
    }

    void allreduce_min_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_min_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_min_long failed");
        }
    }

    // Max reductions
    void allreduce_max(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_max failed");
        }
    }

    void allreduce_max_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_max_int failed");
        }
    }

    void allreduce_max_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_max_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_max_long failed");
        }
    }

    // Product reductions
    void allreduce_prod(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_prod failed");
        }
    }

    void allreduce_prod_int(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_int_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_prod_int failed");
        }
    }

    void allreduce_prod_long(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_prod_long_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_prod_long failed");
        }
    }

    // ------------------------------------------------------------------------
    // Template Convenience Reduction Methods
    // ------------------------------------------------------------------------

    template <typename T>
    T allreduce_sum_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_sum(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    template <typename T>
    T allreduce_min_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_min_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_min_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_min(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_min(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_min_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_min(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    template <typename T>
    T allreduce_max_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_max_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_max_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_max(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_max(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_max_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_max(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Product Reduction (template convenience)
    // ------------------------------------------------------------------------

    template <typename T>
    T allreduce_prod_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            allreduce_prod_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            allreduce_prod_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            allreduce_prod(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, float>) {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_prod(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            allreduce_prod_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            allreduce_prod(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Reduce to Root (template convenience)
    // ------------------------------------------------------------------------

    template <typename T>
    T reduce_sum_to_root(T local, rank_t root) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            auto r = impl_->reduce_sum_int_impl(&local, &result, 1, root);
            if (!r) throw communication_error("NCCL reduce_sum_to_root (int) failed");
        } else if constexpr (std::is_same_v<T, double>) {
            auto r = impl_->reduce_sum_impl(&local, &result, 1, root);
            if (!r) throw communication_error("NCCL reduce_sum_to_root (double) failed");
        } else if constexpr (std::is_integral_v<T>) {
            int i_local = static_cast<int>(local);
            int i_result{};
            auto r = impl_->reduce_sum_int_impl(&i_local, &i_result, 1, root);
            if (!r) throw communication_error("NCCL reduce_sum_to_root (integral) failed");
            result = static_cast<T>(i_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            auto r = impl_->reduce_sum_impl(&d_local, &d_result, 1, root);
            if (!r) throw communication_error("NCCL reduce_sum_to_root (double cast) failed");
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Logical Reductions (emulated via NCCL min/max on int)
    // ------------------------------------------------------------------------

    void allreduce_land(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_land_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_land failed");
        }
    }

    void allreduce_lor(const void* sendbuf, void* recvbuf, size_type count) {
        auto result = impl_->allreduce_lor_impl(sendbuf, recvbuf, count);
        if (!result) {
            throw communication_error("NCCL allreduce_lor failed");
        }
    }

    bool allreduce_land_value(bool local) {
        int i_local = local ? 1 : 0;
        int i_result{};
        allreduce_land(&i_local, &i_result, 1);
        return i_result != 0;
    }

    bool allreduce_lor_value(bool local) {
        int i_local = local ? 1 : 0;
        int i_result{};
        allreduce_lor(&i_local, &i_result, 1);
        return i_result != 0;
    }

    // ------------------------------------------------------------------------
    // Scan Operations (emulated via allgather + local prefix sum)
    // ------------------------------------------------------------------------

    void scan_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        scan_sum_typed<int>(sendbuf, recvbuf, count);
    }

    void scan_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        scan_sum_typed<long>(sendbuf, recvbuf, count);
    }

    void scan_sum(const void* sendbuf, void* recvbuf, size_type count) {
        scan_sum_typed<double>(sendbuf, recvbuf, count);
    }

    void exscan_sum_int(const void* sendbuf, void* recvbuf, size_type count) {
        exscan_sum_typed<int>(sendbuf, recvbuf, count);
    }

    void exscan_sum_long(const void* sendbuf, void* recvbuf, size_type count) {
        exscan_sum_typed<long>(sendbuf, recvbuf, count);
    }

    void exscan_sum(const void* sendbuf, void* recvbuf, size_type count) {
        exscan_sum_typed<double>(sendbuf, recvbuf, count);
    }

    template <typename T>
    T scan_sum_value(T local) {
        T result{};
        if constexpr (std::is_same_v<T, int>) {
            scan_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            scan_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            scan_sum(&local, &result, 1);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            scan_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            scan_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    template <typename T>
    T exscan_sum_value(T local) {
        if (rank() == 0) {
            // Rank 0 gets identity value, but still must participate
            T result{};
            if constexpr (std::is_same_v<T, int>) {
                exscan_sum_int(&local, &result, 1);
            } else if constexpr (std::is_same_v<T, long>) {
                exscan_sum_long(&local, &result, 1);
            } else if constexpr (std::is_same_v<T, double>) {
                exscan_sum(&local, &result, 1);
            } else if constexpr (std::is_integral_v<T>) {
                long l_local = static_cast<long>(local);
                long l_result{};
                exscan_sum_long(&l_local, &l_result, 1);
            } else {
                double d_local = static_cast<double>(local);
                double d_result{};
                exscan_sum(&d_local, &d_result, 1);
            }
            return T{};  // Rank 0 always gets identity
        }

        T result{};
        if constexpr (std::is_same_v<T, int>) {
            exscan_sum_int(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, long>) {
            exscan_sum_long(&local, &result, 1);
        } else if constexpr (std::is_same_v<T, double>) {
            exscan_sum(&local, &result, 1);
        } else if constexpr (std::is_integral_v<T>) {
            long l_local = static_cast<long>(local);
            long l_result{};
            exscan_sum_long(&l_local, &l_result, 1);
            result = static_cast<T>(l_result);
        } else {
            double d_local = static_cast<double>(local);
            double d_result{};
            exscan_sum(&d_local, &d_result, 1);
            result = static_cast<T>(d_result);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Variable-Size Collective Operations
    // ------------------------------------------------------------------------

    void gatherv(const void* sendbuf, size_type sendcount,
                 void* recvbuf, const int* recvcounts, const int* displs,
                 size_type elem_size, rank_t root) {
        auto result = impl_->gatherv_impl(sendbuf, sendcount, recvbuf,
                                          recvcounts, displs, elem_size, root);
        if (!result) {
            throw communication_error("NCCL gatherv failed");
        }
    }

    void scatterv(const void* sendbuf, const int* sendcounts, const int* displs,
                  void* recvbuf, size_type recvcount, size_type elem_size, rank_t root) {
        auto result = impl_->scatterv_impl(sendbuf, sendcounts, displs,
                                           recvbuf, recvcount, elem_size, root);
        if (!result) {
            throw communication_error("NCCL scatterv failed");
        }
    }

    void allgatherv(const void* sendbuf, size_type sendcount,
                    void* recvbuf, const int* recvcounts, const int* displs,
                    size_type elem_size) {
        auto result = impl_->allgatherv_impl(sendbuf, sendcount, recvbuf,
                                             recvcounts, displs, elem_size);
        if (!result) {
            throw communication_error("NCCL allgatherv failed");
        }
    }

    void alltoallv(const void* sendbuf, const int* sendcounts, const int* sdispls,
                   void* recvbuf, const int* recvcounts, const int* rdispls,
                   size_type elem_size) {
        auto result = impl_->alltoallv_impl(sendbuf, sendcounts, sdispls,
                                            recvbuf, recvcounts, rdispls, elem_size);
        if (!result) {
            throw communication_error("NCCL alltoallv failed");
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
    /// @brief Inclusive scan emulated via allgather + local prefix sum
    template <typename T>
    void scan_sum_typed(const void* sendbuf, void* recvbuf, size_type count) {
        size_type n = size();
        std::vector<T> all_data(count * n);
        allgather(sendbuf, all_data.data(), count * sizeof(T));

        // Synchronize to ensure allgather data is available
        auto sync_result = impl_->synchronize();
        (void)sync_result;

        T* out = static_cast<T*>(recvbuf);
        for (size_type i = 0; i < count; ++i) {
            T running{};
            for (rank_t r = 0; r <= rank(); ++r) {
                running += all_data[static_cast<size_type>(r) * count + i];
            }
            out[i] = running;
        }
    }

    /// @brief Exclusive scan emulated via allgather + local prefix sum
    template <typename T>
    void exscan_sum_typed(const void* sendbuf, void* recvbuf, size_type count) {
        size_type n = size();
        std::vector<T> all_data(count * n);
        allgather(sendbuf, all_data.data(), count * sizeof(T));

        // Synchronize to ensure allgather data is available
        auto sync_result = impl_->synchronize();
        (void)sync_result;

        T* out = static_cast<T*>(recvbuf);
        for (size_type i = 0; i < count; ++i) {
            T running{};
            for (rank_t r = 0; r < rank(); ++r) {
                running += all_data[static_cast<size_type>(r) * count + i];
            }
            out[i] = running;
        }
    }

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
