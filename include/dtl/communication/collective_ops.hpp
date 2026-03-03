// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file collective_ops.hpp
/// @brief Collective communication operations
/// @details Provides barrier, broadcast, scatter, gather, reduce, allreduce.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/backend/concepts/communicator.hpp>
#include <dtl/communication/reduction_ops.hpp>

#include <algorithm>
#include <span>
#include <vector>

namespace dtl {

// ============================================================================
// Synchronization
// ============================================================================

/// @brief Synchronize all ranks (collective barrier)
/// @tparam Comm Communicator type satisfying CollectiveCommunicator
/// @param comm The communicator
/// @return Result indicating success or error
template <CollectiveCommunicator Comm>
result<void> barrier(Comm& comm) {
    comm.barrier();
    return {};
}

// ============================================================================
// Broadcast
// ============================================================================

/// @brief Broadcast data from root to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param data Data buffer (input at root, output at others)
/// @param root Root rank that sends the data
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> broadcast(Comm& comm, std::span<T> data, rank_t root = 0) {
    comm.broadcast(data.data(), data.size() * sizeof(T), root);
    return {};
}

/// @brief Broadcast a single value from root to all ranks
/// @tparam Comm Communicator type
/// @tparam T Value type
/// @param comm The communicator
/// @param value Value to broadcast (input at root, output at others)
/// @param root Root rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> broadcast(Comm& comm, T& value, rank_t root = 0) {
    comm.broadcast(&value, sizeof(T), root);
    return {};
}

// ============================================================================
// Scatter
// ============================================================================

/// @brief Scatter data from root to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to scatter (only used at root)
/// @param recv_data Buffer for received data
/// @param root Root rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> scatter(Comm& comm, std::span<const T> send_data,
                     std::span<T> recv_data, rank_t root = 0) {
    comm.scatter(send_data.data(), recv_data.data(),
                 recv_data.size() * sizeof(T), root);
    return {};
}

/// @brief Scatter with varying counts to each rank
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to scatter
/// @param send_counts Count for each rank
/// @param displacements Offset for each rank in send_data
/// @param recv_data Buffer for received data
/// @param root Root rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> scatterv(Comm& comm, std::span<const T> send_data,
                      std::span<const size_type> send_counts,
                      std::span<const size_type> displacements,
                      std::span<T> recv_data, rank_t root = 0) {
    // Convert size_type to int for MPI compatibility
    std::vector<int> int_counts(send_counts.size());
    std::vector<int> int_displs(displacements.size());
    for (size_t i = 0; i < send_counts.size(); ++i) {
        int_counts[i] = static_cast<int>(send_counts[i]);
        int_displs[i] = static_cast<int>(displacements[i]);
    }
    comm.scatterv(send_data.data(), int_counts.data(), int_displs.data(),
                  recv_data.data(), recv_data.size(), sizeof(T), root);
    return {};
}

// ============================================================================
// Gather
// ============================================================================

/// @brief Gather data from all ranks to root
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param recv_data Buffer for gathered data (only used at root)
/// @param root Root rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> gather(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data, rank_t root = 0) {
    comm.gather(send_data.data(), recv_data.data(),
                send_data.size() * sizeof(T), root);
    return {};
}

/// @brief Gather with varying counts from each rank
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param recv_data Buffer for gathered data
/// @param recv_counts Count from each rank
/// @param displacements Offset for each rank in recv_data
/// @param root Root rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> gatherv(Comm& comm, std::span<const T> send_data,
                     std::span<T> recv_data,
                     std::span<const size_type> recv_counts,
                     std::span<const size_type> displacements,
                     rank_t root = 0) {
    // Convert size_type to int for MPI compatibility
    std::vector<int> int_counts(recv_counts.size());
    std::vector<int> int_displs(displacements.size());
    for (size_t i = 0; i < recv_counts.size(); ++i) {
        int_counts[i] = static_cast<int>(recv_counts[i]);
        int_displs[i] = static_cast<int>(displacements[i]);
    }
    comm.gatherv(send_data.data(), send_data.size(),
                 recv_data.data(), int_counts.data(), int_displs.data(),
                 sizeof(T), root);
    return {};
}

/// @brief Gather data from all ranks to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param recv_data Buffer for gathered data
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> allgather(Comm& comm, std::span<const T> send_data,
                       std::span<T> recv_data) {
    comm.allgather(send_data.data(), recv_data.data(),
                   send_data.size() * sizeof(T));
    return {};
}

/// @brief Allgather with varying counts
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param recv_data Buffer for gathered data
/// @param recv_counts Count from each rank
/// @param displacements Offset for each rank
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> allgatherv(Comm& comm, std::span<const T> send_data,
                        std::span<T> recv_data,
                        std::span<const size_type> recv_counts,
                        std::span<const size_type> displacements) {
    // Convert size_type to int for MPI compatibility
    std::vector<int> int_counts(recv_counts.size());
    std::vector<int> int_displs(displacements.size());
    for (size_t i = 0; i < recv_counts.size(); ++i) {
        int_counts[i] = static_cast<int>(recv_counts[i]);
        int_displs[i] = static_cast<int>(displacements[i]);
    }
    comm.allgatherv(send_data.data(), send_data.size(),
                    recv_data.data(), int_counts.data(), int_displs.data(),
                    sizeof(T));
    return {};
}

// ============================================================================
// All-to-All
// ============================================================================

/// @brief Exchange data between all pairs of ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send (chunk i goes to rank i)
/// @param recv_data Buffer for received data
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> alltoall(Comm& comm, std::span<const T> send_data,
                      std::span<T> recv_data) {
    size_type chunk_size = send_data.size() / static_cast<size_type>(comm.size());
    comm.alltoall(send_data.data(), recv_data.data(), chunk_size * sizeof(T));
    return {};
}

/// @brief All-to-all with varying counts
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to send
/// @param send_counts Count to each rank
/// @param send_displacements Offset for each rank in send_data
/// @param recv_data Buffer for received data
/// @param recv_counts Count from each rank
/// @param recv_displacements Offset for each rank in recv_data
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T>
result<void> alltoallv(Comm& comm, std::span<const T> send_data,
                       std::span<const size_type> send_counts,
                       std::span<const size_type> send_displacements,
                       std::span<T> recv_data,
                       std::span<const size_type> recv_counts,
                       std::span<const size_type> recv_displacements) {
    // Convert size_type to int for MPI compatibility
    std::vector<int> int_send_counts(send_counts.size());
    std::vector<int> int_send_displs(send_displacements.size());
    std::vector<int> int_recv_counts(recv_counts.size());
    std::vector<int> int_recv_displs(recv_displacements.size());
    for (size_t i = 0; i < send_counts.size(); ++i) {
        int_send_counts[i] = static_cast<int>(send_counts[i]);
        int_send_displs[i] = static_cast<int>(send_displacements[i]);
        int_recv_counts[i] = static_cast<int>(recv_counts[i]);
        int_recv_displs[i] = static_cast<int>(recv_displacements[i]);
    }
    comm.alltoallv(send_data.data(), int_send_counts.data(), int_send_displs.data(),
                   recv_data.data(), int_recv_counts.data(), int_recv_displs.data(),
                   sizeof(T));
    return {};
}

// ============================================================================
// Reduce Operations
// ============================================================================

/// @brief Reduce data to root using an operation
/// @tparam Comm Communicator type satisfying ReducingCommunicator
/// @tparam T Element type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param send_data Data to reduce
/// @param recv_data Buffer for result (only used at root)
/// @param op Reduction operation
/// @param root Root rank
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T, typename Op>
result<void> reduce(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data, [[maybe_unused]] Op op, rank_t root = 0) {
    // Dispatch based on operation tag
    using Tag = typename Op::tag_type;
    if constexpr (std::is_same_v<Tag, reduce_sum_tag>) {
        // Type-dispatched reduce sum
        if constexpr (std::is_same_v<T, int>) {
            comm.reduce_sum_int(send_data.data(), recv_data.data(),
                                send_data.size(), root);
        } else if constexpr (std::is_same_v<T, long>) {
            // Use int variant with long-sized memcpy for null_communicator,
            // or dedicated long variant if available
            comm.reduce_sum(send_data.data(), recv_data.data(),
                            send_data.size(), root, sizeof(T));
        } else if constexpr (std::is_same_v<T, double>) {
            comm.reduce_sum(send_data.data(), recv_data.data(),
                            send_data.size(), root);
        } else if constexpr (std::is_same_v<T, float>) {
            comm.reduce_sum(send_data.data(), recv_data.data(),
                            send_data.size(), root, sizeof(T));
        } else {
            // Generic fallback: use element-size-aware overload
            comm.reduce_sum(send_data.data(), recv_data.data(),
                            send_data.size(), root, sizeof(T));
        }
    } else if constexpr (std::is_same_v<Tag, reduce_min_tag>) {
        // Element-wise min reduction across ranks
        for (size_t i = 0; i < send_data.size(); ++i) {
            T global_min = comm.template allreduce_min_value<T>(send_data[i]);
            if (comm.rank() == root) {
                recv_data[i] = global_min;
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_max_tag>) {
        // Element-wise max reduction across ranks
        for (size_t i = 0; i < send_data.size(); ++i) {
            T global_max = comm.template allreduce_max_value<T>(send_data[i]);
            if (comm.rank() == root) {
                recv_data[i] = global_max;
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_product_tag>) {
        // Element-wise product reduction across ranks
        for (size_t i = 0; i < send_data.size(); ++i) {
            T global_prod = comm.template allreduce_prod_value<T>(send_data[i]);
            if (comm.rank() == root) {
                recv_data[i] = global_prod;
            }
        }
    } else {
        // Unsupported operation — report error instead of silent fallback
        return make_error<void>(status_code::not_supported,
            "Unsupported reduction operation for reduce");
    }
    return {};
}

/// @brief Reduce a single value to root
/// @tparam Comm Communicator type
/// @tparam T Value type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param value Value to reduce
/// @param op Reduction operation
/// @param root Root rank
/// @return Result at root, undefined at others
template <ReducingCommunicator Comm, typename T, typename Op>
result<T> reduce(Comm& comm, const T& value, Op op, rank_t root = 0) {
    T send_val = value;
    T recv_val{};
    auto res = reduce(comm, std::span<const T>(&send_val, 1),
                      std::span<T>(&recv_val, 1), op, root);
    if (!res) return make_error(res.error().code(), res.error().message());
    return recv_val;
}

/// @brief Reduce data to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param send_data Data to reduce
/// @param recv_data Buffer for result
/// @param op Reduction operation
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T, typename Op>
result<void> allreduce(Comm& comm, std::span<const T> send_data,
                       std::span<T> recv_data, [[maybe_unused]] Op op) {
    using Tag = typename Op::tag_type;
    if constexpr (std::is_same_v<Tag, reduce_sum_tag>) {
        // Type-dispatched allreduce sum
        if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_sum_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_sum_long(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, double>) {
            comm.allreduce_sum(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, float>) {
            // Convert float to double for MPI
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_sum(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        } else if constexpr (std::is_integral_v<T>) {
            // Other integral types via long
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_sum_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        } else {
            // Other types via double
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_sum(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_min_tag>) {
        // Vectorized min: single MPI call for all elements
        if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_min_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_min_long(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, double>) {
            comm.allreduce_min(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, float>) {
            // Convert float to double for MPI
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_min(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        } else if constexpr (std::is_integral_v<T>) {
            // Other integral types via long
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_min_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        } else {
            // Other types via double
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_min(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_max_tag>) {
        // Vectorized max: single MPI call for all elements
        if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_max_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_max_long(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, double>) {
            comm.allreduce_max(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, float>) {
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_max(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        } else if constexpr (std::is_integral_v<T>) {
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_max_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        } else {
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_max(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_product_tag>) {
        // Vectorized product: single MPI call for all elements
        if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_prod_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_prod_long(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, double>) {
            comm.allreduce_prod(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, float>) {
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_prod(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        } else if constexpr (std::is_integral_v<T>) {
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_prod_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        } else {
            std::vector<double> send_double(send_data.begin(), send_data.end());
            std::vector<double> recv_double(recv_data.size());
            comm.allreduce_prod(send_double.data(), recv_double.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_double[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_land_tag>) {
        // Vectorized logical AND: single MPI call for all elements
        std::vector<int> send_int(send_data.size());
        std::vector<int> recv_int(recv_data.size());
        for (size_t i = 0; i < send_data.size(); ++i) {
            send_int[i] = static_cast<bool>(send_data[i]) ? 1 : 0;
        }
        comm.allreduce_land(send_int.data(), recv_int.data(), send_data.size());
        for (size_t i = 0; i < recv_data.size(); ++i) {
            recv_data[i] = static_cast<T>(recv_int[i] != 0);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_lor_tag>) {
        // Vectorized logical OR: single MPI call for all elements
        std::vector<int> send_int(send_data.size());
        std::vector<int> recv_int(recv_data.size());
        for (size_t i = 0; i < send_data.size(); ++i) {
            send_int[i] = static_cast<bool>(send_data[i]) ? 1 : 0;
        }
        comm.allreduce_lor(send_int.data(), recv_int.data(), send_data.size());
        for (size_t i = 0; i < recv_data.size(); ++i) {
            recv_data[i] = static_cast<T>(recv_int[i] != 0);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_band_tag>) {
        // Bitwise AND: requires integral type
        static_assert(std::is_integral_v<T>,
            "Bitwise AND reduction requires an integral type");
        if constexpr (std::is_same_v<T, bool>) {
            // Boolean: map to logical AND
            std::vector<int> send_int(send_data.size());
            std::vector<int> recv_int(recv_data.size());
            for (size_t i = 0; i < send_data.size(); ++i) {
                send_int[i] = send_data[i] ? 1 : 0;
            }
            comm.allreduce_land(send_int.data(), recv_int.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_int[i] != 0);
            }
        } else if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_band_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_band_long(send_data.data(), recv_data.data(), send_data.size());
        } else {
            // Other integral types via long
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_band_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_bor_tag>) {
        // Bitwise OR: requires integral type
        static_assert(std::is_integral_v<T>,
            "Bitwise OR reduction requires an integral type");
        if constexpr (std::is_same_v<T, bool>) {
            // Boolean: map to logical OR
            std::vector<int> send_int(send_data.size());
            std::vector<int> recv_int(recv_data.size());
            for (size_t i = 0; i < send_data.size(); ++i) {
                send_int[i] = send_data[i] ? 1 : 0;
            }
            comm.allreduce_lor(send_int.data(), recv_int.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_int[i] != 0);
            }
        } else if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_bor_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_bor_long(send_data.data(), recv_data.data(), send_data.size());
        } else {
            // Other integral types via long
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_bor_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        }
    } else if constexpr (std::is_same_v<Tag, reduce_bxor_tag>) {
        // Bitwise XOR: requires integral type
        static_assert(std::is_integral_v<T>,
            "Bitwise XOR reduction requires an integral type");
        if constexpr (std::is_same_v<T, bool>) {
            // Boolean: map to logical XOR (a != b)
            std::vector<int> send_int(send_data.size());
            std::vector<int> recv_int(recv_data.size());
            for (size_t i = 0; i < send_data.size(); ++i) {
                send_int[i] = send_data[i] ? 1 : 0;
            }
            comm.allreduce_bxor_int(send_int.data(), recv_int.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_int[i] != 0);
            }
        } else if constexpr (std::is_same_v<T, int>) {
            comm.allreduce_bxor_int(send_data.data(), recv_data.data(), send_data.size());
        } else if constexpr (std::is_same_v<T, long>) {
            comm.allreduce_bxor_long(send_data.data(), recv_data.data(), send_data.size());
        } else {
            // Other integral types via long
            std::vector<long> send_long(send_data.begin(), send_data.end());
            std::vector<long> recv_long(recv_data.size());
            comm.allreduce_bxor_long(send_long.data(), recv_long.data(), send_data.size());
            for (size_t i = 0; i < recv_data.size(); ++i) {
                recv_data[i] = static_cast<T>(recv_long[i]);
            }
        }
    } else {
        // Unsupported operation — report error instead of silent fallback
        return make_error<void>(status_code::not_supported,
            "Unsupported reduction operation for allreduce");
    }
    return {};
}

/// @brief Reduce a single value to all ranks
/// @tparam Comm Communicator type
/// @tparam T Value type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param value Value to reduce
/// @param op Reduction operation
/// @return Reduced value on all ranks
template <ReducingCommunicator Comm, typename T, typename Op>
result<T> allreduce(Comm& comm, const T& value, Op op) {
    T send_val = value;
    T recv_val{};
    auto res = allreduce(comm, std::span<const T>(&send_val, 1),
                         std::span<T>(&recv_val, 1), op);
    if (!res) return make_error(res.error().code(), res.error().message());
    return recv_val;
}

/// @brief In-place allreduce
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param data Data to reduce (in-place)
/// @param op Reduction operation
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T, typename Op>
result<void> allreduce_inplace(Comm& comm, std::span<T> data, Op op) {
    std::vector<T> temp(data.begin(), data.end());
    return allreduce(comm, std::span<const T>(temp), data, op);
}

// ============================================================================
// Scan Operations
// ============================================================================

/// @brief Inclusive prefix scan
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param send_data Data to scan
/// @param recv_data Buffer for result
/// @param op Reduction operation
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T, typename Op>
result<void> scan(Comm& comm, std::span<const T> send_data,
                  std::span<T> recv_data, [[maybe_unused]] Op op) {
    using Tag = typename Op::tag_type;
    if constexpr (std::is_same_v<Tag, reduce_sum_tag>) {
        // Use MPI scan for sum
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template scan_sum_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_min_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template scan_min_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_max_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template scan_max_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_product_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template scan_prod_value<T>(send_data[i]);
        }
    } else {
        // Unsupported operation — report error instead of silent fallback
        return make_error<void>(status_code::not_supported,
            "Unsupported reduction operation for scan");
    }
    return {};
}

/// @brief Exclusive prefix scan
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @tparam Op Reduction operation
/// @param comm The communicator
/// @param send_data Data to scan
/// @param recv_data Buffer for result
/// @param op Reduction operation
/// @return Result indicating success or error
template <CollectiveCommunicator Comm, typename T, typename Op>
result<void> exscan(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data, [[maybe_unused]] Op op) {
    using Tag = typename Op::tag_type;
    if constexpr (std::is_same_v<Tag, reduce_sum_tag>) {
        // Use MPI exscan for sum
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template exscan_sum_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_min_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template exscan_min_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_max_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template exscan_max_value<T>(send_data[i]);
        }
    } else if constexpr (std::is_same_v<Tag, reduce_product_tag>) {
        for (size_t i = 0; i < send_data.size(); ++i) {
            recv_data[i] = comm.template exscan_prod_value<T>(send_data[i]);
        }
    } else {
        // Unsupported operation — report error instead of silent fallback
        return make_error<void>(status_code::not_supported,
            "Unsupported reduction operation for exscan");
    }
    return {};
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// @brief Sum reduction to root
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to sum
/// @param recv_data Buffer for result
/// @param root Root rank
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T>
result<void> sum(Comm& comm, std::span<const T> send_data,
                 std::span<T> recv_data, rank_t root = 0) {
    return reduce(comm, send_data, recv_data, reduce_sum<T>{}, root);
}

/// @brief Sum reduction to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to sum
/// @param recv_data Buffer for result
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T>
result<void> allsum(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data) {
    return allreduce(comm, send_data, recv_data, reduce_sum<T>{});
}

/// @brief Max reduction to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to max
/// @param recv_data Buffer for result
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T>
result<void> allmax(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data) {
    return allreduce(comm, send_data, recv_data, reduce_max<T>{});
}

/// @brief Min reduction to all ranks
/// @tparam Comm Communicator type
/// @tparam T Element type
/// @param comm The communicator
/// @param send_data Data to min
/// @param recv_data Buffer for result
/// @return Result indicating success or error
template <ReducingCommunicator Comm, typename T>
result<void> allmin(Comm& comm, std::span<const T> send_data,
                    std::span<T> recv_data) {
    return allreduce(comm, send_data, recv_data, reduce_min<T>{});
}

}  // namespace dtl
