// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file hierarchical_communicator.hpp
/// @brief Hierarchical communicator combining NCCL intra-node + MPI inter-node
/// @details Provides optimized communication using NCCL for GPU-to-GPU
///          within a node and MPI for inter-node transfers.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#if DTL_ENABLE_NCCL
#include <nccl.h>
#endif

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#include <cstring>
#include <memory>
#include <vector>

namespace dtl {
namespace hybrid {

// ============================================================================
// Communication Level
// ============================================================================

/// @brief Level of communication hierarchy
enum class comm_level {
    /// @brief Within same GPU (no-op)
    intra_gpu,

    /// @brief Between GPUs on same node (NCCL)
    intra_node,

    /// @brief Between nodes (MPI)
    inter_node,

    /// @brief Automatic selection based on topology
    automatic
};

// ============================================================================
// Hierarchical Topology
// ============================================================================

/// @brief Node-level topology information
struct node_topology {
    /// @brief World rank
    rank_t world_rank = no_rank;

    /// @brief Total world size
    rank_t world_size = 0;

    /// @brief Node index (0-based)
    size_type node_index = 0;

    /// @brief Total number of nodes
    size_type num_nodes = 0;

    /// @brief Rank within node
    rank_t local_rank = 0;

    /// @brief Number of ranks (GPUs) on this node
    rank_t local_size = 0;

    /// @brief Whether this rank is the node leader
    bool is_node_leader() const noexcept { return local_rank == 0; }
};

// ============================================================================
// Hierarchical Communicator
// ============================================================================

// ---------------------------------------------------------------------------
// BETA BACKEND
// ---------------------------------------------------------------------------
// Status:   Beta — core hierarchical operations implemented
// Supports: barrier, broadcast, allreduce (two-level NCCL + MPI)
// Missing:  - GPUDirect RDMA inter-node data path
//           - Host-staging fallback for non-GPUDirect systems
// Requires: DTL_ENABLE_MPI, DTL_ENABLE_NCCL, DTL_ENABLE_CUDA at build time.
// ---------------------------------------------------------------------------

/// @brief Hierarchical communicator for hybrid MPI + NCCL communication
/// @details Uses NCCL for efficient intra-node GPU communication and
///          MPI for inter-node communication. Automatically selects
///          the optimal communication path based on rank topology.
///
///          Two-level collective pattern:
///          1. NCCL within each node (all local GPUs)
///          2. MPI among node leaders (one per node)
///          3. NCCL broadcast of result within each node
class hierarchical_communicator : public communicator_base {
public:
    /// @brief Configuration for hierarchical communicator
    struct config {
        /// @brief Use GPUDirect RDMA if available
        bool enable_gpu_direct = true;

        /// @brief Stage through host for inter-node if GPUDirect unavailable
        bool enable_host_staging = true;

        /// @brief Size of host staging buffer per rank
        size_type staging_buffer_size = 64 * 1024 * 1024;  // 64 MB
    };

    /// @brief Default constructor
    hierarchical_communicator() = default;

#if DTL_ENABLE_MPI
    /// @brief Construct from MPI communicator with default config
    /// @param world_comm MPI world communicator
    explicit hierarchical_communicator(MPI_Comm world_comm)
        : config_() {
        initialize(world_comm);
    }

    /// @brief Construct from MPI communicator with custom config
    /// @param world_comm MPI world communicator
    /// @param cfg Configuration options
    hierarchical_communicator(MPI_Comm world_comm, const config& cfg)
        : config_(cfg) {
        initialize(world_comm);
    }
#endif

    /// @brief Destructor
    ~hierarchical_communicator() override {
        cleanup();
    }

    // Non-copyable
    hierarchical_communicator(const hierarchical_communicator&) = delete;
    hierarchical_communicator& operator=(const hierarchical_communicator&) = delete;

    // Movable
    hierarchical_communicator(hierarchical_communicator&& other) noexcept
        : topology_(other.topology_)
        , config_(other.config_)
#if DTL_ENABLE_MPI
        , world_comm_(other.world_comm_)
        , node_comm_(other.node_comm_)
        , leader_comm_(other.leader_comm_)
#endif
#if DTL_ENABLE_NCCL
        , nccl_comm_(other.nccl_comm_)
#endif
#if DTL_ENABLE_CUDA
        , stream_(other.stream_)
        , staging_buf_(other.staging_buf_)
#endif
    {
#if DTL_ENABLE_MPI
        other.world_comm_ = MPI_COMM_NULL;
        other.node_comm_ = MPI_COMM_NULL;
        other.leader_comm_ = MPI_COMM_NULL;
#endif
#if DTL_ENABLE_NCCL
        other.nccl_comm_ = nullptr;
#endif
#if DTL_ENABLE_CUDA
        other.stream_ = nullptr;
        other.staging_buf_ = nullptr;
#endif
    }

    hierarchical_communicator& operator=(hierarchical_communicator&&) = default;

    // ------------------------------------------------------------------------
    // Communicator Interface
    // ------------------------------------------------------------------------

    [[nodiscard]] rank_t rank() const noexcept override {
        return topology_.world_rank;
    }

    [[nodiscard]] rank_t size() const noexcept override {
        return topology_.world_size;
    }

    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_MPI
        return world_comm_ != MPI_COMM_NULL;
#else
        return false;
#endif
    }

    /// @brief Get communicator properties
    [[nodiscard]] communicator_properties properties() const noexcept override {
        return communicator_properties{
            .size = topology_.world_size,
            .rank = topology_.world_rank,
            .is_inter = false,
            .is_derived = false,
            .name = "hierarchical"
        };
    }

    // ------------------------------------------------------------------------
    // Point-to-Point Communication
    // ------------------------------------------------------------------------

    result<void> send_impl(const void* data, size_type count,
                          size_type elem_size, rank_t dest, int tag) {
        // Determine communication level
        auto level = determine_level(topology_.world_rank, dest);

        switch (level) {
            case comm_level::intra_gpu:
                return {};  // Same GPU, no-op

            case comm_level::intra_node:
                return send_intra_node(data, count * elem_size, dest, tag);

            case comm_level::inter_node:
                return send_inter_node(data, count * elem_size, dest, tag);

            default:
                return send_auto(data, count * elem_size, dest, tag);
        }
    }

    result<void> recv_impl(void* data, size_type count,
                          size_type elem_size, rank_t source, int tag) {
        auto level = determine_level(source, topology_.world_rank);

        switch (level) {
            case comm_level::intra_gpu:
                return {};

            case comm_level::intra_node:
                return recv_intra_node(data, count * elem_size, source, tag);

            case comm_level::inter_node:
                return recv_inter_node(data, count * elem_size, source, tag);

            default:
                return recv_auto(data, count * elem_size, source, tag);
        }
    }

    // ------------------------------------------------------------------------
    // Collective Communication
    // ------------------------------------------------------------------------

    /// @brief Two-phase barrier: NCCL intra-node + MPI inter-node
    result<void> barrier() {
        auto r = barrier_intra_node();
        if (!r) return r;
        return barrier_inter_node();
    }

    /// @brief Hierarchical broadcast
    /// @details 1. Root broadcasts to node leaders via MPI
    ///          2. Each node leader broadcasts within node via NCCL
    result<void> broadcast_impl(void* data, size_type count,
                               size_type elem_size, rank_t root) {
        size_type total_size = count * elem_size;

        auto r = broadcast_inter_node(data, total_size, root);
        if (!r) return r;

        return broadcast_intra_node(data, total_size, 0);  // Local root = 0
    }

    /// @brief Gather using MPI fallback (not hierarchical two-level)
    /// @details Gather requires ordered assembly of data from all ranks,
    ///          which makes a true two-level NCCL+MPI decomposition
    ///          non-trivial. This implementation falls back to flat MPI
    ///          gather with host staging when CUDA is enabled. GPU data
    ///          is first copied to a host buffer, MPI_Gather is called on
    ///          the world communicator, and the result is copied back to
    ///          device memory on the root rank.
    result<void> gather_impl(const void* send_data, size_type send_count,
                            void* recv_data, size_type recv_count,
                            size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        (void)send_count; (void)recv_count;
        size_type send_bytes = send_count * elem_size;
        size_type recv_bytes = recv_count * elem_size;

        // Stage GPU data to host for MPI gather
        std::vector<char> host_send(send_bytes);
        std::vector<char> host_recv;
        if (topology_.world_rank == root) {
            host_recv.resize(static_cast<size_type>(topology_.world_size) * recv_bytes);
        }

#if DTL_ENABLE_CUDA
        cudaMemcpy(host_send.data(), send_data, send_bytes, cudaMemcpyDeviceToHost);
#else
        std::memcpy(host_send.data(), send_data, send_bytes);
#endif

        MPI_Gather(host_send.data(), static_cast<int>(send_bytes), MPI_BYTE,
                   host_recv.data(), static_cast<int>(recv_bytes), MPI_BYTE,
                   root, world_comm_);

        if (topology_.world_rank == root) {
#if DTL_ENABLE_CUDA
            cudaMemcpy(recv_data, host_recv.data(), host_recv.size(),
                       cudaMemcpyHostToDevice);
#else
            std::memcpy(recv_data, host_recv.data(), host_recv.size());
#endif
        }
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "MPI not enabled for hierarchical gather");
#endif
    }

    /// @brief Scatter using MPI fallback (not hierarchical two-level)
    /// @details Like gather_impl, scatter requires ordered distribution of
    ///          data to all ranks. This implementation falls back to flat
    ///          MPI scatter with host staging when CUDA is enabled. On the
    ///          root rank, GPU data is first copied to a host buffer, then
    ///          MPI_Scatter distributes it, and each rank copies its portion
    ///          back to device memory.
    result<void> scatter_impl(const void* send_data, size_type send_count,
                             void* recv_data, size_type recv_count,
                             size_type elem_size, rank_t root) {
#if DTL_ENABLE_MPI
        size_type send_bytes = send_count * elem_size;
        size_type recv_bytes = recv_count * elem_size;

        std::vector<char> host_send;
        if (topology_.world_rank == root) {
            host_send.resize(static_cast<size_type>(topology_.world_size) * send_bytes);
#if DTL_ENABLE_CUDA
            cudaMemcpy(host_send.data(), send_data,
                       host_send.size(), cudaMemcpyDeviceToHost);
#else
            std::memcpy(host_send.data(), send_data, host_send.size());
#endif
        }

        std::vector<char> host_recv(recv_bytes);
        MPI_Scatter(host_send.data(), static_cast<int>(send_bytes), MPI_BYTE,
                    host_recv.data(), static_cast<int>(recv_bytes), MPI_BYTE,
                    root, world_comm_);

#if DTL_ENABLE_CUDA
        cudaMemcpy(recv_data, host_recv.data(), recv_bytes, cudaMemcpyHostToDevice);
#else
        std::memcpy(recv_data, host_recv.data(), recv_bytes);
#endif
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "MPI not enabled for hierarchical scatter");
#endif
    }

    result<void> allgather_impl(const void* send_data, size_type send_count,
                               void* recv_data, size_type recv_count,
                               size_type elem_size) {
#if DTL_ENABLE_MPI
        size_type send_bytes = send_count * elem_size;
        size_type recv_bytes = recv_count * elem_size;

        std::vector<char> host_send(send_bytes);
        std::vector<char> host_recv(static_cast<size_type>(topology_.world_size) * recv_bytes);

#if DTL_ENABLE_CUDA
        cudaMemcpy(host_send.data(), send_data, send_bytes, cudaMemcpyDeviceToHost);
#else
        std::memcpy(host_send.data(), send_data, send_bytes);
#endif

        MPI_Allgather(host_send.data(), static_cast<int>(send_bytes), MPI_BYTE,
                      host_recv.data(), static_cast<int>(recv_bytes), MPI_BYTE,
                      world_comm_);

#if DTL_ENABLE_CUDA
        cudaMemcpy(recv_data, host_recv.data(), host_recv.size(),
                   cudaMemcpyHostToDevice);
#else
        std::memcpy(recv_data, host_recv.data(), host_recv.size());
#endif
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "MPI not enabled for hierarchical allgather");
#endif
    }

    // ------------------------------------------------------------------------
    // Hierarchical Reductions
    // ------------------------------------------------------------------------

    /// @brief Hierarchical allreduce
    /// @tparam T Data type
    /// @param send_buf Send buffer (device memory)
    /// @param recv_buf Receive buffer (device memory)
    /// @param count Number of elements
    /// @return Success or error
    template <typename T>
    result<void> allreduce(const T* send_buf, T* recv_buf, size_type count) {
        // Two-phase allreduce:
        // 1. Reduce within node using NCCL
        // 2. Allreduce among node leaders using MPI
        // 3. Broadcast result within node

        auto r = reduce_intra_node(send_buf, recv_buf, count);
        if (!r) return r;

        if (topology_.is_node_leader()) {
            r = allreduce_inter_node(recv_buf, count);
            if (!r) return r;
        }

        return broadcast_intra_node(recv_buf, count * sizeof(T), 0);
    }

    // ------------------------------------------------------------------------
    // Topology Access
    // ------------------------------------------------------------------------

    /// @brief Get topology information
    [[nodiscard]] const node_topology& topology() const noexcept {
        return topology_;
    }

    /// @brief Determine communication level between two ranks
    [[nodiscard]] comm_level determine_level(rank_t src, rank_t dst) const noexcept {
        if (src == dst) return comm_level::intra_gpu;

        // Check if same node using local_size-based partitioning
        rank_t src_node = (topology_.local_size > 0) ? src / topology_.local_size : 0;
        rank_t dst_node = (topology_.local_size > 0) ? dst / topology_.local_size : 0;

        return (src_node == dst_node) ? comm_level::intra_node
                                      : comm_level::inter_node;
    }

    // ------------------------------------------------------------------------
    // Native Handle Access
    // ------------------------------------------------------------------------

#if DTL_ENABLE_MPI
    [[nodiscard]] MPI_Comm world_comm() const noexcept { return world_comm_; }
    [[nodiscard]] MPI_Comm node_comm() const noexcept { return node_comm_; }
    [[nodiscard]] MPI_Comm leader_comm() const noexcept { return leader_comm_; }
#endif

#if DTL_ENABLE_NCCL
    [[nodiscard]] ncclComm_t nccl_comm() const noexcept { return nccl_comm_; }
#endif

private:
#if DTL_ENABLE_MPI
    void initialize(MPI_Comm world_comm) {
        world_comm_ = world_comm;

        MPI_Comm_rank(world_comm_, &topology_.world_rank);
        MPI_Comm_size(world_comm_, &topology_.world_size);

        // Create intra-node communicator via shared memory type split
        MPI_Comm_split_type(world_comm_, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &node_comm_);
        MPI_Comm_rank(node_comm_, &topology_.local_rank);
        MPI_Comm_size(node_comm_, &topology_.local_size);

        // Create inter-node leader communicator (only local_rank==0 participates)
        int color = (topology_.local_rank == 0) ? 0 : MPI_UNDEFINED;
        MPI_Comm_split(world_comm_, color, topology_.world_rank, &leader_comm_);

        // Calculate node index
        if (topology_.local_rank == 0) {
            int leader_rank;
            MPI_Comm_rank(leader_comm_, &leader_rank);
            topology_.node_index = static_cast<size_type>(leader_rank);
        }
        MPI_Bcast(&topology_.node_index, 1, MPI_UNSIGNED_LONG, 0, node_comm_);

        // Count nodes
        if (topology_.local_rank == 0) {
            int num_leaders;
            MPI_Comm_size(leader_comm_, &num_leaders);
            topology_.num_nodes = static_cast<size_type>(num_leaders);
        }
        MPI_Bcast(&topology_.num_nodes, 1, MPI_UNSIGNED_LONG, 0, node_comm_);

        // Allocate host staging buffer for inter-node transfers
#if DTL_ENABLE_CUDA
        if (config_.enable_host_staging) {
            cudaMallocHost(&staging_buf_, config_.staging_buffer_size);
        }
#endif

        // Initialize NCCL
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        initialize_nccl();
#endif
    }
#endif

#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
    void initialize_nccl() {
        ncclUniqueId id;
        if (topology_.local_rank == 0) {
            ncclGetUniqueId(&id);
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, node_comm_);

        cudaStreamCreate(&stream_);
        ncclCommInitRank(&nccl_comm_, topology_.local_size,
                         id, topology_.local_rank);
    }
#endif

    void cleanup() {
#if DTL_ENABLE_NCCL
        if (nccl_comm_) {
            ncclCommDestroy(nccl_comm_);
            nccl_comm_ = nullptr;
        }
#endif

#if DTL_ENABLE_CUDA
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (staging_buf_) {
            cudaFreeHost(staging_buf_);
            staging_buf_ = nullptr;
        }
#endif

#if DTL_ENABLE_MPI
        if (leader_comm_ != MPI_COMM_NULL && leader_comm_ != MPI_COMM_WORLD) {
            MPI_Comm_free(&leader_comm_);
        }
        if (node_comm_ != MPI_COMM_NULL) {
            MPI_Comm_free(&node_comm_);
        }
        // Don't free world_comm_ - we don't own it
#endif
    }

    // ========================================================================
    // Intra-Node Operations (NCCL)
    // ========================================================================

    result<void> barrier_intra_node() {
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        if (!nccl_comm_ || !stream_) return {};
        // NCCL has no barrier — use allreduce on a single int as barrier
        int dummy = 0;
        int* d_dummy = nullptr;
        cudaError_t cerr = cudaMalloc(&d_dummy, sizeof(int));
        if (cerr != cudaSuccess) {
            return make_error<void>(status_code::backend_error,
                                   "cudaMalloc failed in intra-node barrier");
        }
        cudaMemcpy(d_dummy, &dummy, sizeof(int), cudaMemcpyHostToDevice);
        ncclResult_t nres = ncclAllReduce(d_dummy, d_dummy, 1, ncclInt, ncclSum,
                                          nccl_comm_, stream_);
        cudaStreamSynchronize(stream_);
        cudaFree(d_dummy);
        if (nres != ncclSuccess) {
            return make_error<void>(status_code::barrier_failed,
                                   "NCCL intra-node barrier failed");
        }
        return {};
#elif DTL_ENABLE_MPI
        if (node_comm_ != MPI_COMM_NULL) {
            MPI_Barrier(node_comm_);
        }
        return {};
#else
        return {};
#endif
    }

    result<void> barrier_inter_node() {
#if DTL_ENABLE_MPI
        if (topology_.is_node_leader() && leader_comm_ != MPI_COMM_NULL) {
            MPI_Barrier(leader_comm_);
        }
        // Synchronize within node so all ranks wait for leader
        if (node_comm_ != MPI_COMM_NULL) {
            MPI_Barrier(node_comm_);
        }
        return {};
#else
        return {};
#endif
    }

    result<void> broadcast_intra_node(void* data, size_type total_size, rank_t local_root) {
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        if (!nccl_comm_ || !stream_) return {};
        ncclResult_t nres = ncclBroadcast(data, data, total_size, ncclChar,
                                          local_root, nccl_comm_, stream_);
        cudaStreamSynchronize(stream_);
        if (nres != ncclSuccess) {
            return make_error<void>(status_code::collective_failure,
                                   "NCCL intra-node broadcast failed");
        }
        return {};
#elif DTL_ENABLE_MPI
        if (node_comm_ != MPI_COMM_NULL) {
            MPI_Bcast(data, static_cast<int>(total_size), MPI_BYTE,
                      local_root, node_comm_);
        }
        (void)data; (void)total_size; (void)local_root;
        return {};
#else
        (void)data; (void)total_size; (void)local_root;
        return {};
#endif
    }

    result<void> broadcast_inter_node(void* data, size_type total_size, rank_t root) {
#if DTL_ENABLE_MPI
        // Determine which node leader corresponds to root
        // Root's node leader is the one that broadcasts
        rank_t root_leader = root / ((topology_.local_size > 0) ? topology_.local_size : 1);

        if (topology_.is_node_leader() && leader_comm_ != MPI_COMM_NULL) {
            // Stage from GPU to host if needed
#if DTL_ENABLE_CUDA
            if (staging_buf_ && total_size <= config_.staging_buffer_size) {
                if (topology_.world_rank == root) {
                    cudaMemcpy(staging_buf_, data, total_size, cudaMemcpyDeviceToHost);
                }
                MPI_Bcast(staging_buf_, static_cast<int>(total_size), MPI_BYTE,
                          root_leader, leader_comm_);
                if (topology_.world_rank != root) {
                    cudaMemcpy(data, staging_buf_, total_size, cudaMemcpyHostToDevice);
                }
            } else {
                // Direct MPI (assumes GPU-aware MPI or small enough for stack)
                MPI_Bcast(data, static_cast<int>(total_size), MPI_BYTE,
                          root_leader, leader_comm_);
            }
#else
            MPI_Bcast(data, static_cast<int>(total_size), MPI_BYTE,
                      root_leader, leader_comm_);
#endif
        }
        return {};
#else
        (void)data; (void)total_size; (void)root;
        return {};
#endif
    }

    template <typename T>
    result<void> reduce_intra_node(const T* send_buf, T* recv_buf, size_type count) {
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        if (!nccl_comm_ || !stream_) {
            // Single-node or no NCCL: just copy
            if (send_buf != recv_buf) {
                cudaMemcpy(recv_buf, send_buf, count * sizeof(T),
                           cudaMemcpyDeviceToDevice);
            }
            return {};
        }
        // Use ncclReduce to node leader (local rank 0)
        ncclDataType_t dtype = nccl_type_of<T>();
        ncclResult_t nres = ncclReduce(send_buf, recv_buf, count, dtype,
                                       ncclSum, 0, nccl_comm_, stream_);
        cudaStreamSynchronize(stream_);
        if (nres != ncclSuccess) {
            return make_error<void>(status_code::reduce_failed,
                                   "NCCL intra-node reduce failed");
        }
        return {};
#else
        // No NCCL: copy through
        (void)count;
        if (send_buf != recv_buf) {
            std::memcpy(recv_buf, send_buf, count * sizeof(T));
        }
        return {};
#endif
    }

    template <typename T>
    result<void> allreduce_inter_node(T* buf, size_type count) {
#if DTL_ENABLE_MPI
        if (leader_comm_ == MPI_COMM_NULL) return {};
        // Stage GPU data to host for MPI
#if DTL_ENABLE_CUDA
        size_type bytes = count * sizeof(T);
        std::vector<char> host_buf(bytes);
        cudaMemcpy(host_buf.data(), buf, bytes, cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE, host_buf.data(), static_cast<int>(count),
                      mpi_type_of<T>(), MPI_SUM, leader_comm_);
        cudaMemcpy(buf, host_buf.data(), bytes, cudaMemcpyHostToDevice);
#else
        MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(count),
                      mpi_type_of<T>(), MPI_SUM, leader_comm_);
#endif
        return {};
#else
        (void)buf; (void)count;
        return {};
#endif
    }

    // ========================================================================
    // Point-to-Point Helpers
    // ========================================================================

    result<void> send_intra_node(const void* data, size_type bytes,
                                 rank_t dest, int tag) {
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        if (!nccl_comm_ || !stream_) return {};
        // Map world rank to local rank for NCCL send
        rank_t local_dest = dest % topology_.local_size;
        ncclGroupStart();
        ncclSend(data, bytes, ncclChar, local_dest, nccl_comm_, stream_);
        ncclGroupEnd();
        cudaStreamSynchronize(stream_);
        (void)tag;
        return {};
#elif DTL_ENABLE_MPI
        MPI_Send(data, static_cast<int>(bytes), MPI_BYTE, dest, tag, world_comm_);
        return {};
#else
        (void)data; (void)bytes; (void)dest; (void)tag;
        return {};
#endif
    }

    result<void> send_inter_node(const void* data, size_type bytes,
                                  rank_t dest, int tag) {
#if DTL_ENABLE_MPI
#if DTL_ENABLE_CUDA
        // Stage from GPU to host, then MPI send
        if (staging_buf_ && bytes <= config_.staging_buffer_size) {
            cudaMemcpy(staging_buf_, data, bytes, cudaMemcpyDeviceToHost);
            MPI_Send(staging_buf_, static_cast<int>(bytes), MPI_BYTE,
                     dest, tag, world_comm_);
        } else {
            // Assume GPU-aware MPI
            MPI_Send(data, static_cast<int>(bytes), MPI_BYTE,
                     dest, tag, world_comm_);
        }
#else
        MPI_Send(data, static_cast<int>(bytes), MPI_BYTE, dest, tag, world_comm_);
#endif
        return {};
#else
        (void)data; (void)bytes; (void)dest; (void)tag;
        return {};
#endif
    }

    result<void> send_auto(const void* data, size_type bytes, rank_t dest, int tag) {
        auto level = determine_level(topology_.world_rank, dest);
        if (level == comm_level::intra_node)
            return send_intra_node(data, bytes, dest, tag);
        return send_inter_node(data, bytes, dest, tag);
    }

    result<void> recv_intra_node(void* data, size_type bytes,
                                  rank_t source, int tag) {
#if DTL_ENABLE_NCCL && DTL_ENABLE_CUDA
        if (!nccl_comm_ || !stream_) return {};
        rank_t local_src = source % topology_.local_size;
        ncclGroupStart();
        ncclRecv(data, bytes, ncclChar, local_src, nccl_comm_, stream_);
        ncclGroupEnd();
        cudaStreamSynchronize(stream_);
        (void)tag;
        return {};
#elif DTL_ENABLE_MPI
        MPI_Recv(data, static_cast<int>(bytes), MPI_BYTE, source, tag,
                 world_comm_, MPI_STATUS_IGNORE);
        return {};
#else
        (void)data; (void)bytes; (void)source; (void)tag;
        return {};
#endif
    }

    result<void> recv_inter_node(void* data, size_type bytes,
                                  rank_t source, int tag) {
#if DTL_ENABLE_MPI
#if DTL_ENABLE_CUDA
        if (staging_buf_ && bytes <= config_.staging_buffer_size) {
            MPI_Recv(staging_buf_, static_cast<int>(bytes), MPI_BYTE,
                     source, tag, world_comm_, MPI_STATUS_IGNORE);
            cudaMemcpy(data, staging_buf_, bytes, cudaMemcpyHostToDevice);
        } else {
            MPI_Recv(data, static_cast<int>(bytes), MPI_BYTE,
                     source, tag, world_comm_, MPI_STATUS_IGNORE);
        }
#else
        MPI_Recv(data, static_cast<int>(bytes), MPI_BYTE, source, tag,
                 world_comm_, MPI_STATUS_IGNORE);
#endif
        return {};
#else
        (void)data; (void)bytes; (void)source; (void)tag;
        return {};
#endif
    }

    result<void> recv_auto(void* data, size_type bytes, rank_t source, int tag) {
        auto level = determine_level(source, topology_.world_rank);
        if (level == comm_level::intra_node)
            return recv_intra_node(data, bytes, source, tag);
        return recv_inter_node(data, bytes, source, tag);
    }

    // ========================================================================
    // Type Mapping Helpers
    // ========================================================================

#if DTL_ENABLE_NCCL
    template <typename T>
    static ncclDataType_t nccl_type_of() {
        if constexpr (std::is_same_v<T, float>) return ncclFloat32;
        else if constexpr (std::is_same_v<T, double>) return ncclFloat64;
        else if constexpr (std::is_same_v<T, int32_t>) return ncclInt32;
        else if constexpr (std::is_same_v<T, int64_t>) return ncclInt64;
        else if constexpr (std::is_same_v<T, uint32_t>) return ncclUint32;
        else if constexpr (std::is_same_v<T, uint64_t>) return ncclUint64;
        else return ncclChar;
    }
#endif

#if DTL_ENABLE_MPI
    template <typename T>
    static MPI_Datatype mpi_type_of() {
        if constexpr (std::is_same_v<T, float>) return MPI_FLOAT;
        else if constexpr (std::is_same_v<T, double>) return MPI_DOUBLE;
        else if constexpr (std::is_same_v<T, int32_t>) return MPI_INT32_T;
        else if constexpr (std::is_same_v<T, int64_t>) return MPI_INT64_T;
        else if constexpr (std::is_same_v<T, uint32_t>) return MPI_UINT32_T;
        else if constexpr (std::is_same_v<T, uint64_t>) return MPI_UINT64_T;
        else return MPI_BYTE;
    }
#endif

    // ========================================================================
    // Member Data
    // ========================================================================

    node_topology topology_;
    config config_;

#if DTL_ENABLE_MPI
    MPI_Comm world_comm_ = MPI_COMM_NULL;
    MPI_Comm node_comm_ = MPI_COMM_NULL;
    MPI_Comm leader_comm_ = MPI_COMM_NULL;
#endif

#if DTL_ENABLE_NCCL
    ncclComm_t nccl_comm_ = nullptr;
#endif

#if DTL_ENABLE_CUDA
    cudaStream_t stream_ = nullptr;
    void* staging_buf_ = nullptr;
#endif
};

// ============================================================================
// Factory Functions
// ============================================================================

#if DTL_ENABLE_MPI
/// @brief Create a hierarchical communicator from MPI_COMM_WORLD
[[nodiscard]] inline std::unique_ptr<hierarchical_communicator>
make_hierarchical_communicator() {
    return std::make_unique<hierarchical_communicator>(MPI_COMM_WORLD);
}

/// @brief Create a hierarchical communicator from custom MPI comm
[[nodiscard]] inline std::unique_ptr<hierarchical_communicator>
make_hierarchical_communicator(MPI_Comm comm,
                               const hierarchical_communicator::config& cfg) {
    return std::make_unique<hierarchical_communicator>(comm, cfg);
}
#endif

}  // namespace hybrid
}  // namespace dtl
