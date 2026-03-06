// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file domain_impl.hpp
/// @brief Domain implementation details (inline definitions requiring backend headers)
/// @details Include this header after domain.hpp when you need full domain functionality.
///          This header includes backend-specific headers (MPI, CUDA, etc.).
/// @since 0.1.0

#pragma once

#include <dtl/core/domain.hpp>
#include <cstdlib>  // std::abort for stub implementations

#if DTL_ENABLE_MPI
#include <backends/mpi/mpi_comm_adapter.hpp>
#endif

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

#if DTL_ENABLE_NCCL
#include <nccl.h>
#include <backends/nccl/nccl_comm_adapter.hpp>
#endif

namespace dtl {

// =============================================================================
// MPI Domain Implementation
// =============================================================================

#if DTL_ENABLE_MPI

inline mpi_domain::mpi_domain()
    : adapter_(std::make_shared<mpi::mpi_comm_adapter>())
    , rank_(adapter_->rank())
    , size_(adapter_->size()) {}

inline mpi_domain::mpi_domain(std::shared_ptr<mpi::mpi_comm_adapter> adapter) noexcept
    : adapter_(std::move(adapter))
    , rank_(adapter_ ? adapter_->rank() : 0)
    , size_(adapter_ ? adapter_->size() : 1) {}

inline rank_t mpi_domain::rank() const noexcept {
    return rank_;
}

inline rank_t mpi_domain::size() const noexcept {
    return size_;
}

inline bool mpi_domain::valid() const noexcept {
    return adapter_ != nullptr && size_ > 0 && rank_ != no_rank;
}

inline bool mpi_domain::is_root() const noexcept {
    return rank_ == 0;
}

inline mpi::mpi_comm_adapter& mpi_domain::communicator() noexcept {
    return *adapter_;
}

inline const mpi::mpi_comm_adapter& mpi_domain::communicator() const noexcept {
    return *adapter_;
}

inline std::shared_ptr<mpi::mpi_comm_adapter> mpi_domain::adapter_handle() const noexcept {
    return adapter_;
}

inline result<mpi_domain> mpi_domain::split(int color, int key) const {
    if (!valid()) {
        return result<mpi_domain>::failure(
            status{status_code::invalid_state, no_rank, "Cannot split invalid MPI domain"});
    }

    try {
        auto split_adapter = adapter_->split(color, key);
        return result<mpi_domain>::success(
            mpi_domain(std::make_shared<mpi::mpi_comm_adapter>(std::move(split_adapter))));
    } catch (const std::exception& e) {
        return result<mpi_domain>::failure(
            status{status_code::operation_failed, no_rank, std::string("MPI split failed: ") + e.what()});
    }
}

inline void mpi_domain::barrier() {
    if (valid()) {
        adapter_->barrier();
    }
}

// Concept verification for MPI domain
static_assert(CommunicationDomain<mpi_domain>,
              "mpi_domain must satisfy CommunicationDomain concept");

#else  // !DTL_ENABLE_MPI

// Stub implementations when MPI is disabled
inline mpi_domain::mpi_domain() = default;

inline mpi_domain::mpi_domain(std::shared_ptr<mpi::mpi_comm_adapter>) noexcept {}

inline rank_t mpi_domain::rank() const noexcept { return 0; }
inline rank_t mpi_domain::size() const noexcept { return 1; }
inline bool mpi_domain::valid() const noexcept { return false; }
inline bool mpi_domain::is_root() const noexcept { return true; }

// Note: These methods should never be called when MPI is disabled.
// They exist only to satisfy the interface. Calling them is undefined behavior.
inline mpi::mpi_comm_adapter& mpi_domain::communicator() noexcept {
    // This should never be called when MPI is disabled
    std::abort();
}

inline const mpi::mpi_comm_adapter& mpi_domain::communicator() const noexcept {
    // This should never be called when MPI is disabled
    std::abort();
}

inline std::shared_ptr<mpi::mpi_comm_adapter> mpi_domain::adapter_handle() const noexcept {
    return nullptr;
}

inline result<mpi_domain> mpi_domain::split(int, int) const {
    return result<mpi_domain>::failure(
        status{status_code::not_supported, no_rank, "MPI not enabled"});
}

inline void mpi_domain::barrier() {}

#endif  // DTL_ENABLE_MPI

// =============================================================================
// CUDA Domain Implementation
// =============================================================================

#if DTL_ENABLE_CUDA

inline cuda_domain::cuda_domain() {
    cudaError_t err = cudaGetDevice(&device_id_);
    if (err == cudaSuccess && device_id_ >= 0) {
        err = cudaStreamCreate(&stream_);
        if (err == cudaSuccess) {
            valid_ = true;
            owns_stream_ = true;
        }
    }
}

inline cuda_domain::cuda_domain(int device_id) : device_id_(device_id) {
    if (device_id_ >= 0) {
        cudaError_t err = cudaSetDevice(device_id_);
        if (err == cudaSuccess) {
            err = cudaStreamCreate(&stream_);
            if (err == cudaSuccess) {
                valid_ = true;
                owns_stream_ = true;
            }
        }
    }
}

inline cuda_domain::cuda_domain(int device_id, cudaStream_t stream)
    : device_id_(device_id)
    , stream_(stream)
    , valid_(device_id >= 0 && stream != nullptr)
    , owns_stream_(false) {}

inline void cuda_domain::synchronize() {
    if (valid_ && stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }
}

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// HIP Domain Implementation
// =============================================================================

#if DTL_ENABLE_HIP

inline hip_domain::hip_domain() {
    hipError_t err = hipGetDevice(&device_id_);
    if (err == hipSuccess && device_id_ >= 0) {
        err = hipStreamCreate(&stream_);
        if (err == hipSuccess) {
            valid_ = true;
            owns_stream_ = true;
        }
    }
}

inline hip_domain::hip_domain(int device_id) : device_id_(device_id) {
    if (device_id_ >= 0) {
        hipError_t err = hipSetDevice(device_id_);
        if (err == hipSuccess) {
            err = hipStreamCreate(&stream_);
            if (err == hipSuccess) {
                valid_ = true;
                owns_stream_ = true;
            }
        }
    }
}

inline void hip_domain::synchronize() {
    if (valid_ && stream_ != nullptr) {
        hipStreamSynchronize(stream_);
    }
}

#endif  // DTL_ENABLE_HIP

// =============================================================================
// SHMEM Domain Implementation
// =============================================================================

#if DTL_ENABLE_SHMEM

inline shmem_domain::shmem_domain() {
    // Query SHMEM runtime state
    rank_ = shmem_my_pe();
    size_ = shmem_n_pes();
    valid_ = (size_ > 0);
}

inline void shmem_domain::barrier() {
    if (valid_) {
        shmem_barrier_all();
    }
}

#endif  // DTL_ENABLE_SHMEM

// =============================================================================
// NCCL Domain Implementation
// =============================================================================

#if DTL_ENABLE_NCCL

#if DTL_ENABLE_CUDA
#include <backends/nccl/nccl_communicator.hpp>
#endif

inline nccl_domain::nccl_domain(std::shared_ptr<nccl::nccl_communicator> comm) noexcept
    : comm_(std::move(comm)) {
    if (comm_) {
        rank_ = comm_->rank();
        size_ = comm_->size();
        adapter_ = std::make_shared<nccl::nccl_comm_adapter>(comm_);
    }
}

inline rank_t nccl_domain::rank() const noexcept {
    return rank_;
}

inline rank_t nccl_domain::size() const noexcept {
    return size_;
}

inline bool nccl_domain::valid() const noexcept {
    return comm_ != nullptr && comm_->valid();
}

inline bool nccl_domain::is_root() const noexcept {
    return rank_ == 0;
}

inline nccl::nccl_comm_adapter& nccl_domain::adapter() noexcept {
    return *adapter_;
}

inline const nccl::nccl_comm_adapter& nccl_domain::adapter() const noexcept {
    return *adapter_;
}

inline result<nccl_domain> nccl_domain::from_mpi(const mpi_domain& mpi, int device_id) {
    if (!mpi.valid()) {
        return result<nccl_domain>::failure(
            status{status_code::invalid_state, no_rank,
                   "Cannot create NCCL domain from invalid MPI domain"});
    }

#if DTL_ENABLE_CUDA && DTL_ENABLE_MPI
    // Verify CUDA is available and device_id is valid
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        return result<nccl_domain>::failure(
            status{status_code::backend_error, no_rank,
                   "CUDA not available or no CUDA devices found"});
    }

    if (device_id < 0 || device_id >= device_count) {
        return result<nccl_domain>::failure(
            status{status_code::invalid_argument, no_rank,
                   "Invalid CUDA device ID for NCCL domain creation"});
    }

    // Set the device before any NCCL operations
    cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        return result<nccl_domain>::failure(
            status{status_code::backend_error, no_rank,
                   "Failed to set CUDA device for NCCL domain"});
    }

    // Get MPI rank and size for the NCCL communicator
    rank_t mpi_rank = mpi.rank();
    rank_t mpi_size = mpi.size();

    // Step 1: Rank 0 generates the unique ID
    ncclUniqueId unique_id;
    if (mpi_rank == 0) {
        auto id_result = nccl::get_unique_id();
        if (!id_result) {
            // Broadcast failure to all ranks (use negative size as signal)
            int failure_flag = -1;
            MPI_Bcast(&failure_flag, 1, MPI_INT, 0,
                      const_cast<mpi::mpi_comm_adapter&>(mpi.communicator()).underlying().native_handle());
            return result<nccl_domain>::failure(id_result.error());
        }
        unique_id = *id_result;
    }

    // Step 2: Broadcast the unique ID from rank 0 to all ranks
    // First broadcast a success flag
    int success_flag = (mpi_rank == 0) ? 1 : 0;
    MPI_Bcast(&success_flag, 1, MPI_INT, 0,
              const_cast<mpi::mpi_comm_adapter&>(mpi.communicator()).underlying().native_handle());

    if (success_flag < 0) {
        return result<nccl_domain>::failure(
            status{status_code::communication_error, no_rank,
                   "NCCL unique ID generation failed on rank 0"});
    }

    // Broadcast the actual unique ID
    MPI_Bcast(&unique_id, sizeof(ncclUniqueId), MPI_BYTE, 0,
              const_cast<mpi::mpi_comm_adapter&>(mpi.communicator()).underlying().native_handle());

    // Step 3: Initialize NCCL communicator on all ranks
    auto comm_result = nccl::create_communicator_from_unique_id(
        unique_id, mpi_rank, mpi_size, device_id);

    if (!comm_result) {
        return result<nccl_domain>::failure(comm_result.error());
    }

    return result<nccl_domain>::success(nccl_domain(std::move(*comm_result)));
#else
    (void)device_id;
    return result<nccl_domain>::failure(
        status{status_code::not_supported, no_rank,
               "NCCL domain creation requires both CUDA and MPI backends"});
#endif  // DTL_ENABLE_CUDA && DTL_ENABLE_MPI
}

inline result<std::pair<mpi_domain, nccl_domain>>
nccl_domain::split(const mpi_domain& mpi, int color, int device_id, int key) {
    // Step 1: Split the MPI communicator
    auto mpi_split = mpi.split(color, key);
    if (!mpi_split) {
        return result<std::pair<mpi_domain, nccl_domain>>::failure(mpi_split.error());
    }

    // Step 2: Create new NCCL domain from the split MPI sub-communicator
    auto nccl_result = nccl_domain::from_mpi(*mpi_split, device_id);
    if (!nccl_result) {
        return result<std::pair<mpi_domain, nccl_domain>>::failure(nccl_result.error());
    }

    return result<std::pair<mpi_domain, nccl_domain>>::success(
        std::make_pair(std::move(*mpi_split), std::move(*nccl_result)));
}

#endif  // DTL_ENABLE_NCCL

}  // namespace dtl
