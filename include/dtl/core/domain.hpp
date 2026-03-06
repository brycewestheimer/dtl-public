// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file domain.hpp
/// @brief Domain abstractions for multi-domain context architecture
/// @details Provides domain types that wrap backend resources (MPI, CUDA, NCCL, SHMEM, CPU)
///          for use with the variadic context<Domains...> template.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>

#include <memory>
#include <type_traits>
#include <utility>

#if DTL_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

#if DTL_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace dtl {

// =============================================================================
// Domain Tags
// =============================================================================

/// @brief Tag type for MPI communication domain
struct mpi_domain_tag {};

/// @brief Tag type for NCCL communication domain
struct nccl_domain_tag {};

/// @brief Tag type for SHMEM communication domain
struct shmem_domain_tag {};

/// @brief Tag type for CPU execution domain
struct cpu_domain_tag {};

/// @brief Tag type for CUDA execution domain
struct cuda_domain_tag {};

/// @brief Tag type for HIP execution domain
struct hip_domain_tag {};

// =============================================================================
// Domain Concepts
// =============================================================================

/// @brief Concept for communication domain types
/// @details Communication domains provide rank/size queries for distributed computation.
template <typename D>
concept CommunicationDomain = requires(const D& d) {
    { d.rank() } -> std::same_as<rank_t>;
    { d.size() } -> std::same_as<rank_t>;
    { d.valid() } -> std::same_as<bool>;
    typename D::tag_type;
};

/// @brief Concept for execution domain types
/// @details Execution domains provide compute capabilities (CPU, GPU).
template <typename D>
concept ExecutionDomain = requires(const D& d) {
    { d.valid() } -> std::same_as<bool>;
    typename D::tag_type;
};

// =============================================================================
// CPU Domain
// =============================================================================

/// @brief CPU execution domain
/// @details Trivial domain representing single-threaded CPU execution.
///          Always valid, rank=0, size=1.
class cpu_domain {
public:
    using tag_type = cpu_domain_tag;

    /// @brief Default constructor
    constexpr cpu_domain() noexcept = default;

    /// @brief Get rank (always 0 for CPU domain)
    [[nodiscard]] constexpr rank_t rank() const noexcept { return 0; }

    /// @brief Get size (always 1 for CPU domain)
    [[nodiscard]] constexpr rank_t size() const noexcept { return 1; }

    /// @brief Check if domain is valid (always true)
    [[nodiscard]] constexpr bool valid() const noexcept { return true; }

    /// @brief Check if this is the root (always true)
    [[nodiscard]] constexpr bool is_root() const noexcept { return true; }
};

static_assert(CommunicationDomain<cpu_domain>,
              "cpu_domain must satisfy CommunicationDomain concept");
static_assert(ExecutionDomain<cpu_domain>,
              "cpu_domain must satisfy ExecutionDomain concept");

// =============================================================================
// MPI Domain
// =============================================================================

// Forward declarations for MPI types
namespace mpi {
class mpi_comm_adapter;
class mpi_communicator;
mpi_communicator& world_communicator();
}  // namespace mpi

/// @brief MPI communication domain
/// @details Wraps an MPI communicator adapter for distributed communication.
///          Uses shared_ptr for safe lifetime management when splitting.
class mpi_domain {
public:
    using tag_type = mpi_domain_tag;

    /// @brief Default constructor (wraps MPI_COMM_WORLD)
    mpi_domain();

    /// @brief Construct from existing MPI communicator adapter
    /// @param adapter Shared pointer to MPI adapter (takes ownership)
    explicit mpi_domain(std::shared_ptr<mpi::mpi_comm_adapter> adapter) noexcept;

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept;

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept;

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept;

    /// @brief Check if this is the root rank (rank 0)
    [[nodiscard]] bool is_root() const noexcept;

    /// @brief Get the underlying communicator adapter
    [[nodiscard]] mpi::mpi_comm_adapter& communicator() noexcept;

    /// @brief Get the underlying communicator adapter (const)
    [[nodiscard]] const mpi::mpi_comm_adapter& communicator() const noexcept;

    /// @brief Get shared ownership handle for the underlying adapter
    [[nodiscard]] std::shared_ptr<mpi::mpi_comm_adapter> adapter_handle() const noexcept;

    /// @brief Split communicator by color
    /// @param color Color for grouping (ranks with same color in same group)
    /// @param key Ordering key within color group (default 0)
    /// @return Result containing new mpi_domain with split communicator
    [[nodiscard]] result<mpi_domain> split(int color, int key = 0) const;

    /// @brief Barrier synchronization
    void barrier();

private:
    std::shared_ptr<mpi::mpi_comm_adapter> adapter_;
    rank_t rank_{0};
    rank_t size_{1};
};

// =============================================================================
// CUDA Domain
// =============================================================================

#if DTL_ENABLE_CUDA

/// @brief CUDA execution domain
/// @details Wraps CUDA device selection and stream management.
class cuda_domain {
public:
    using tag_type = cuda_domain_tag;

    /// @brief Default constructor (uses current device)
    cuda_domain();

    /// @brief Construct with specific device ID
    /// @param device_id CUDA device ID
    explicit cuda_domain(int device_id);

    /// @brief Construct with device ID and existing stream
    /// @param device_id CUDA device ID
    /// @param stream CUDA stream (not owned)
    cuda_domain(int device_id, cudaStream_t stream);

    /// @brief Destructor — destroys owned stream
    ~cuda_domain() {
        if (owns_stream_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Non-copyable (stream ownership)
    cuda_domain(const cuda_domain&) = delete;
    cuda_domain& operator=(const cuda_domain&) = delete;

    // Movable
    cuda_domain(cuda_domain&& other) noexcept
        : device_id_(other.device_id_)
        , stream_(other.stream_)
        , valid_(other.valid_)
        , owns_stream_(other.owns_stream_) {
        other.stream_ = nullptr;
        other.owns_stream_ = false;
        other.valid_ = false;
    }

    cuda_domain& operator=(cuda_domain&& other) noexcept {
        if (this != &other) {
            if (owns_stream_ && stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            device_id_ = other.device_id_;
            stream_ = other.stream_;
            valid_ = other.valid_;
            owns_stream_ = other.owns_stream_;
            other.stream_ = nullptr;
            other.owns_stream_ = false;
            other.valid_ = false;
        }
        return *this;
    }

    /// @brief Get device ID
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    /// @brief Get CUDA stream
    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept { return valid_; }

    /// @brief Synchronize the stream
    void synchronize();

private:
    int device_id_{-1};
    cudaStream_t stream_{nullptr};
    bool valid_{false};
    bool owns_stream_{false};
};

static_assert(ExecutionDomain<cuda_domain>,
              "cuda_domain must satisfy ExecutionDomain concept");

#else  // !DTL_ENABLE_CUDA

/// @brief CUDA domain stub (when CUDA is disabled)
class cuda_domain {
public:
    using tag_type = cuda_domain_tag;

    cuda_domain() = default;
    explicit cuda_domain(int) {}

    [[nodiscard]] int device_id() const noexcept { return -1; }
    [[nodiscard]] bool valid() const noexcept { return false; }
    void synchronize() {}
};

#endif  // DTL_ENABLE_CUDA

// =============================================================================
// NCCL Domain
// =============================================================================

#if DTL_ENABLE_NCCL

// Forward declarations
namespace nccl {
class nccl_communicator;
class nccl_comm_adapter;
}

/// @brief NCCL communication domain
/// @details Wraps NCCL communicator for GPU-accelerated collectives.
class nccl_domain {
public:
    using tag_type = nccl_domain_tag;

    /// @brief Default constructor (invalid domain)
    nccl_domain() noexcept = default;

    /// @brief Construct from existing NCCL communicator
    explicit nccl_domain(std::shared_ptr<nccl::nccl_communicator> comm) noexcept;

    /// @brief Get this process's rank
    [[nodiscard]] rank_t rank() const noexcept;

    /// @brief Get total number of ranks
    [[nodiscard]] rank_t size() const noexcept;

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept;

    /// @brief Check if this is the root rank
    [[nodiscard]] bool is_root() const noexcept;

    /// @brief Get the underlying NCCL communicator
    /// @return Reference to the NCCL communicator
    [[nodiscard]] nccl::nccl_communicator& communicator() noexcept {
        return *comm_;
    }

    /// @brief Get the underlying NCCL communicator (const)
    [[nodiscard]] const nccl::nccl_communicator& communicator() const noexcept {
        return *comm_;
    }

    /// @brief Get the concept-compliant adapter
    /// @details The adapter is limited to explicit device-buffer communication.
    ///          It is not a generic drop-in for host-buffer distributed algorithms.
    [[nodiscard]] nccl::nccl_comm_adapter& adapter() noexcept;

    /// @brief Get the concept-compliant adapter (const)
    /// @details The adapter is limited to explicit device-buffer communication.
    ///          It is not a generic drop-in for host-buffer distributed algorithms.
    [[nodiscard]] const nccl::nccl_comm_adapter& adapter() const noexcept;

    /// @brief Factory: create NCCL domain from MPI domain
    /// @param mpi MPI domain to derive rank/size from
    /// @param device_id CUDA device ID for this rank
    /// @return Result containing new nccl_domain
    [[nodiscard]] static result<nccl_domain> from_mpi(const mpi_domain& mpi, int device_id);

    /// @brief Split NCCL domain via MPI split + new NCCL communicator
    /// @param mpi MPI domain used for bootstrapping the split
    /// @param color Color for grouping (ranks with same color in same group)
    /// @param device_id CUDA device ID for this rank in the new communicator
    /// @param key Ordering key within color group (default 0)
    /// @return Result containing pair of (new mpi_domain, new nccl_domain) for the sub-group
    /// @note This API is currently C++-only. C, Python, and Fortran bindings do
    ///       not expose an equivalent split operation.
    [[nodiscard]] static result<std::pair<mpi_domain, nccl_domain>>
    split(const mpi_domain& mpi, int color, int device_id, int key = 0);

private:
    std::shared_ptr<nccl::nccl_communicator> comm_;
    std::shared_ptr<nccl::nccl_comm_adapter> adapter_;
    rank_t rank_{0};
    rank_t size_{1};
};

#else  // !DTL_ENABLE_NCCL

/// @brief NCCL domain stub (when NCCL is disabled)
class nccl_domain {
public:
    using tag_type = nccl_domain_tag;

    nccl_domain() noexcept = default;

    [[nodiscard]] rank_t rank() const noexcept { return 0; }
    [[nodiscard]] rank_t size() const noexcept { return 1; }
    [[nodiscard]] bool valid() const noexcept { return false; }
    [[nodiscard]] bool is_root() const noexcept { return true; }

    [[nodiscard]] static result<nccl_domain> from_mpi(const mpi_domain&, int) {
        return result<nccl_domain>::failure(
            status{status_code::not_supported, no_rank, "NCCL not enabled"});
    }

    [[nodiscard]] static result<std::pair<mpi_domain, nccl_domain>>
    split(const mpi_domain&, int, int, int = 0) {
        return result<std::pair<mpi_domain, nccl_domain>>::failure(
            status{status_code::not_supported, no_rank, "NCCL not enabled"});
    }
};

#endif  // DTL_ENABLE_NCCL

// =============================================================================
// SHMEM Domain
// =============================================================================

#if DTL_ENABLE_SHMEM

/// @brief OpenSHMEM communication domain
/// @details Wraps SHMEM PE (processing element) queries.
class shmem_domain {
public:
    using tag_type = shmem_domain_tag;

    /// @brief Default constructor (queries SHMEM runtime)
    shmem_domain();

    /// @brief Get this PE's rank
    [[nodiscard]] rank_t rank() const noexcept { return rank_; }

    /// @brief Get total number of PEs
    [[nodiscard]] rank_t size() const noexcept { return size_; }

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept { return valid_; }

    /// @brief Check if this is PE 0 (root)
    [[nodiscard]] bool is_root() const noexcept { return rank_ == 0; }

    /// @brief Barrier synchronization
    void barrier();

private:
    rank_t rank_{0};
    rank_t size_{1};
    bool valid_{false};
};

#else  // !DTL_ENABLE_SHMEM

/// @brief SHMEM domain stub (when SHMEM is disabled)
class shmem_domain {
public:
    using tag_type = shmem_domain_tag;

    shmem_domain() noexcept = default;

    [[nodiscard]] rank_t rank() const noexcept { return 0; }
    [[nodiscard]] rank_t size() const noexcept { return 1; }
    [[nodiscard]] bool valid() const noexcept { return false; }
    [[nodiscard]] bool is_root() const noexcept { return true; }
    void barrier() {}
};

#endif  // DTL_ENABLE_SHMEM

// =============================================================================
// HIP Domain
// =============================================================================

#if DTL_ENABLE_HIP

/// @brief HIP/ROCm execution domain
/// @details Wraps HIP device selection and stream management.
class hip_domain {
public:
    using tag_type = hip_domain_tag;

    /// @brief Default constructor (uses current device)
    hip_domain();

    /// @brief Construct with specific device ID
    explicit hip_domain(int device_id);

    /// @brief Destructor — destroys owned stream
    ~hip_domain() {
        if (owns_stream_ && stream_ != nullptr) {
            hipStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Non-copyable (stream ownership)
    hip_domain(const hip_domain&) = delete;
    hip_domain& operator=(const hip_domain&) = delete;

    // Movable
    hip_domain(hip_domain&& other) noexcept
        : device_id_(other.device_id_)
        , stream_(other.stream_)
        , valid_(other.valid_)
        , owns_stream_(other.owns_stream_) {
        other.stream_ = nullptr;
        other.owns_stream_ = false;
        other.valid_ = false;
    }

    hip_domain& operator=(hip_domain&& other) noexcept {
        if (this != &other) {
            if (owns_stream_ && stream_ != nullptr) {
                hipStreamDestroy(stream_);
            }
            device_id_ = other.device_id_;
            stream_ = other.stream_;
            valid_ = other.valid_;
            owns_stream_ = other.owns_stream_;
            other.stream_ = nullptr;
            other.owns_stream_ = false;
            other.valid_ = false;
        }
        return *this;
    }

    /// @brief Get device ID
    [[nodiscard]] int device_id() const noexcept { return device_id_; }

    /// @brief Get HIP stream
    [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }

    /// @brief Check if domain is valid
    [[nodiscard]] bool valid() const noexcept { return valid_; }

    /// @brief Synchronize the stream
    void synchronize();

private:
    int device_id_{-1};
    hipStream_t stream_{nullptr};
    bool valid_{false};
    bool owns_stream_{false};
};

#else  // !DTL_ENABLE_HIP

/// @brief HIP domain stub (when HIP is disabled)
class hip_domain {
public:
    using tag_type = hip_domain_tag;

    hip_domain() = default;
    explicit hip_domain(int) {}

    [[nodiscard]] int device_id() const noexcept { return -1; }
    [[nodiscard]] bool valid() const noexcept { return false; }
    void synchronize() {}
};

#endif  // DTL_ENABLE_HIP

// =============================================================================
// Domain Type Traits
// =============================================================================

/// @brief Check if type is a communication domain
template <typename T>
struct is_communication_domain : std::bool_constant<CommunicationDomain<T>> {};

template <typename T>
inline constexpr bool is_communication_domain_v = is_communication_domain<T>::value;

/// @brief Check if type is an execution domain
template <typename T>
struct is_execution_domain : std::bool_constant<ExecutionDomain<T>> {};

template <typename T>
inline constexpr bool is_execution_domain_v = is_execution_domain<T>::value;

/// @brief Get the primary communication domain from a domain pack
/// @details Returns the first CommunicationDomain in the pack, or void if none.
template <typename... Domains>
struct primary_communication_domain;

template <>
struct primary_communication_domain<> {
    using type = void;
};

template <typename First, typename... Rest>
struct primary_communication_domain<First, Rest...> {
    using type = std::conditional_t<
        CommunicationDomain<First>,
        First,
        typename primary_communication_domain<Rest...>::type>;
};

template <typename... Domains>
using primary_communication_domain_t = typename primary_communication_domain<Domains...>::type;

}  // namespace dtl
