// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file shmem_communicator.hpp
/// @brief OpenSHMEM communicator for PGAS-style communication
/// @details Provides one-sided communication using OpenSHMEM primitives.
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/error/result.hpp>
#include <dtl/communication/communicator_base.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_SHMEM
#include <shmem.h>
#endif

#include <memory>
#include <vector>

namespace dtl {
namespace shmem {

// ============================================================================
// SHMEM Initialization
// ============================================================================

/// @brief Initialize OpenSHMEM environment
/// @return Success or error
[[nodiscard]] inline result<void> init() {
#if DTL_ENABLE_SHMEM
    shmem_init();
    return {};
#else
    return make_error<void>(status_code::not_supported,
                           "OpenSHMEM support not enabled");
#endif
}

/// @brief Finalize OpenSHMEM environment
inline void finalize() {
#if DTL_ENABLE_SHMEM
    shmem_finalize();
#endif
}

/// @brief Check if SHMEM is initialized
[[nodiscard]] inline bool is_initialized() noexcept {
#if DTL_ENABLE_SHMEM
    // OpenSHMEM doesn't have a query function, assume true after init
    return true;
#else
    return false;
#endif
}

// ============================================================================
// Symmetric Memory Allocation
// ============================================================================

/// @brief Allocate symmetric memory
/// @param size Size in bytes
/// @return Pointer to symmetric memory or error
[[nodiscard]] inline result<void*> symmetric_alloc(size_type size) {
#if DTL_ENABLE_SHMEM
    void* ptr = shmem_malloc(size);
    if (ptr == nullptr) {
        return make_error<void*>(status_code::out_of_memory,
                                "shmem_malloc failed");
    }
    return ptr;
#else
    (void)size;
    return make_error<void*>(status_code::not_supported,
                            "OpenSHMEM support not enabled");
#endif
}

/// @brief Free symmetric memory
/// @param ptr Pointer to symmetric memory
inline void symmetric_free(void* ptr) {
#if DTL_ENABLE_SHMEM
    if (ptr) shmem_free(ptr);
#else
    (void)ptr;
#endif
}

// ============================================================================
// SHMEM Communicator
// ============================================================================

/// @brief OpenSHMEM-based communicator for PGAS-style communication
/// @details Provides one-sided put/get operations and collective primitives.
class shmem_communicator : public communicator_base {
public:
    /// @brief Default constructor
    shmem_communicator() {
#if DTL_ENABLE_SHMEM
        rank_ = shmem_my_pe();
        size_ = shmem_n_pes();
#endif
    }

    /// @brief Destructor
    ~shmem_communicator() override = default;

    // Non-copyable, non-movable (global state)
    shmem_communicator(const shmem_communicator&) = delete;
    shmem_communicator& operator=(const shmem_communicator&) = delete;
    shmem_communicator(shmem_communicator&&) = delete;
    shmem_communicator& operator=(shmem_communicator&&) = delete;

    // ------------------------------------------------------------------------
    // Communicator Interface
    // ------------------------------------------------------------------------

    [[nodiscard]] rank_t rank() const noexcept override { return rank_; }
    [[nodiscard]] rank_t size() const noexcept override { return size_; }

    [[nodiscard]] bool valid() const noexcept {
#if DTL_ENABLE_SHMEM
        return size_ > 0;
#else
        return false;
#endif
    }

    /// @brief Get communicator properties
    [[nodiscard]] communicator_properties properties() const noexcept override {
        return communicator_properties{
            .size = size_,
            .rank = rank_,
            .is_inter = false,
            .is_derived = false,
            .name = "shmem"
        };
    }

    // ------------------------------------------------------------------------
    // Point-to-Point (via put/get emulation)
    // ------------------------------------------------------------------------

    result<void> send_impl(const void* data, size_type count,
                          size_type elem_size, rank_t dest, int tag) {
        // SHMEM uses one-sided, so send = put
        (void)tag;  // SHMEM doesn't use tags
        return put(dest, data, count * elem_size);
    }

    result<void> recv_impl(void* data, size_type count,
                          size_type elem_size, rank_t source, int tag) {
        (void)tag;
        return get(source, data, count * elem_size);
    }

    // ------------------------------------------------------------------------
    // One-Sided Operations (SHMEM native)
    // ------------------------------------------------------------------------

    /// @brief Put data to remote PE (one-sided write)
    /// @param dest Destination PE
    /// @param dest_sym Symmetric destination pointer (on remote PE)
    /// @param src_ptr Local source pointer
    /// @param size Size in bytes
    /// @return Success or error
    /// @note dest_sym must point to symmetric memory allocated with shmem_malloc
    result<void> put(rank_t dest, void* dest_sym, const void* src_ptr, size_type size) {
#if DTL_ENABLE_SHMEM
        shmem_putmem(dest_sym, src_ptr, size, dest);
        return {};
#else
        (void)dest; (void)dest_sym; (void)src_ptr; (void)size;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    /// @brief Get data from remote PE (one-sided read)
    /// @param source Source PE
    /// @param dest_ptr Local destination pointer
    /// @param src_sym Symmetric source pointer (on remote PE)
    /// @param size Size in bytes
    /// @return Success or error
    /// @note src_sym must point to symmetric memory
    result<void> get(rank_t source, void* dest_ptr, const void* src_sym, size_type size) {
#if DTL_ENABLE_SHMEM
        shmem_getmem(dest_ptr, src_sym, size, source);
        return {};
#else
        (void)source; (void)dest_ptr; (void)src_sym; (void)size;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    /// @brief Blocking typed put (int)
    void put_int(int* dest, const int* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_int_put(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Blocking typed put (long)
    void put_long(long* dest, const long* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_long_put(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Blocking typed put (double)
    void put_double(double* dest, const double* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_double_put(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Blocking typed get (int)
    void get_int(int* dest, const int* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_int_get(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Blocking typed get (long)
    void get_long(long* dest, const long* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_long_get(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Blocking typed get (double)
    void get_double(double* dest, const double* source, size_type count, rank_t pe) {
#if DTL_ENABLE_SHMEM
        shmem_double_get(dest, source, count, pe);
#else
        (void)dest; (void)source; (void)count; (void)pe;
#endif
    }

    /// @brief Put data to remote PE (legacy single-pointer version)
    /// @deprecated Use put(dest, dest_sym, src_ptr, size) instead
    result<void> put(rank_t dest, const void* src_ptr, size_type size) {
        (void)dest; (void)src_ptr; (void)size;
        return make_error<void>(status_code::invalid_argument,
                               "Symmetric destination pointer required");
    }

    /// @brief Get data from remote PE (legacy single-pointer version)
    /// @deprecated Use get(source, dest_ptr, src_sym, size) instead
    result<void> get(rank_t source, void* dest_ptr, size_type size) {
        (void)source; (void)dest_ptr; (void)size;
        return make_error<void>(status_code::invalid_argument,
                               "Symmetric source pointer required");
    }

    /// @brief Non-blocking put
    template <typename T>
    result<void> put_nbi(rank_t dest, const T* src, T* dest_sym, size_type count) {
#if DTL_ENABLE_SHMEM
        shmem_putmem_nbi(dest_sym, src, count * sizeof(T), dest);
        return {};
#else
        (void)dest; (void)src; (void)dest_sym; (void)count;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    /// @brief Non-blocking get
    template <typename T>
    result<void> get_nbi(rank_t source, T* dest, const T* src_sym, size_type count) {
#if DTL_ENABLE_SHMEM
        shmem_getmem_nbi(dest, src_sym, count * sizeof(T), source);
        return {};
#else
        (void)source; (void)dest; (void)src_sym; (void)count;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    // ------------------------------------------------------------------------
    // Collective Communication
    // ------------------------------------------------------------------------

    result<void> barrier() {
#if DTL_ENABLE_SHMEM
        shmem_barrier_all();
        return {};
#else
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    result<void> broadcast_impl(void* data, size_type count,
                               size_type elem_size, rank_t root) {
#if DTL_ENABLE_SHMEM
        // SHMEM broadcast (OpenSHMEM 1.5+ team-based API).
        // data must point to symmetric memory allocated via shmem_malloc.
        size_type total_bytes = count * elem_size;
        shmem_broadcastmem(SHMEM_TEAM_WORLD, data, data, total_bytes, root);
        return {};
#else
        (void)data; (void)count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

    result<void> gather_impl(const void* send_data, size_type send_count,
                            void* recv_data, size_type recv_count,
                            size_type elem_size, rank_t root) {
#if DTL_ENABLE_SHMEM
        // SHMEM fcollect gathers equal-sized data from all PEs (allgather).
        // Both send_data and recv_data must be in symmetric memory.
        // Note: SHMEM has no root-specific gather; fcollect is all-to-all.
        (void)recv_count; (void)root;
        size_type send_bytes = send_count * elem_size;
        shmem_fcollectmem(SHMEM_TEAM_WORLD, recv_data, send_data, send_bytes);
        return {};
#else
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "SHMEM collect requires symmetric memory");
#endif
    }

    result<void> scatter_impl(const void* send_data, size_type send_count,
                             void* recv_data, size_type recv_count,
                             size_type elem_size, rank_t root) {
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size; (void)root;
        return make_error<void>(status_code::not_supported,
                               "SHMEM doesn't have native scatter");
    }

    result<void> allgather_impl(const void* send_data, size_type send_count,
                               void* recv_data, size_type recv_count,
                               size_type elem_size) {
        (void)send_data; (void)send_count; (void)recv_data;
        (void)recv_count; (void)elem_size;
        return make_error<void>(status_code::not_supported,
                               "SHMEM fcollect requires symmetric memory");
    }

    // ------------------------------------------------------------------------
    // SHMEM-Specific Operations
    // ------------------------------------------------------------------------

    /// @brief Fence to order remote operations
    void fence() {
#if DTL_ENABLE_SHMEM
        shmem_fence();
#endif
    }

    /// @brief Quiet to wait for all remote operations
    void quiet() {
#if DTL_ENABLE_SHMEM
        shmem_quiet();
#endif
    }

    /// @brief Atomic fetch-and-add
    template <typename T>
    result<T> atomic_fetch_add(T* target, T value, rank_t pe) {
#if DTL_ENABLE_SHMEM
        if constexpr (std::is_same_v<T, int32_t>) {
            return shmem_int_atomic_fetch_add(target, value, pe);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return shmem_long_atomic_fetch_add(target, value, pe);
        }
        return make_error<T>(status_code::not_supported, "Type not supported");
#else
        (void)target; (void)value; (void)pe;
        return make_error<T>(status_code::not_supported,
                            "OpenSHMEM support not enabled");
#endif
    }

    /// @brief Atomic compare-and-swap
    template <typename T>
    result<T> atomic_compare_swap(T* target, T compare, T value, rank_t pe) {
#if DTL_ENABLE_SHMEM
        if constexpr (std::is_same_v<T, int32_t>) {
            return shmem_int_atomic_compare_swap(target, compare, value, pe);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return shmem_long_atomic_compare_swap(target, compare, value, pe);
        }
        return make_error<T>(status_code::not_supported, "Type not supported");
#else
        (void)target; (void)compare; (void)value; (void)pe;
        return make_error<T>(status_code::not_supported,
                            "OpenSHMEM support not enabled");
#endif
    }

    /// @brief Typed reduction (sum)
    /// @details Uses team-based reductions (OpenSHMEM 1.5+).
    ///          Both dest and src must point to symmetric memory.
    template <typename T>
    result<void> reduce_sum(T* dest, const T* src, size_type count) {
#if DTL_ENABLE_SHMEM
        if constexpr (std::is_same_v<T, int>) {
            shmem_int_sum_reduce(SHMEM_TEAM_WORLD, dest, src, count);
        } else if constexpr (std::is_same_v<T, long>) {
            shmem_long_sum_reduce(SHMEM_TEAM_WORLD, dest, src, count);
        } else if constexpr (std::is_same_v<T, long long>) {
            shmem_longlong_sum_reduce(SHMEM_TEAM_WORLD, dest, src, count);
        } else if constexpr (std::is_same_v<T, float>) {
            shmem_float_sum_reduce(SHMEM_TEAM_WORLD, dest, src, count);
        } else if constexpr (std::is_same_v<T, double>) {
            shmem_double_sum_reduce(SHMEM_TEAM_WORLD, dest, src, count);
        } else {
            return make_error<void>(status_code::not_supported,
                                   "Unsupported type for SHMEM sum reduction");
        }
        return {};
#else
        (void)dest; (void)src; (void)count;
        return make_error<void>(status_code::not_supported,
                               "OpenSHMEM support not enabled");
#endif
    }

private:
    rank_t rank_ = no_rank;
    rank_t size_ = 0;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// @brief Get the global SHMEM communicator
[[nodiscard]] inline shmem_communicator& world_communicator() {
    static shmem_communicator comm;
    return comm;
}

// ============================================================================
// RAII Environment Wrapper
// ============================================================================

/// @brief RAII wrapper for SHMEM initialization/finalization
class scoped_shmem_environment {
public:
    scoped_shmem_environment() { init(); }
    ~scoped_shmem_environment() { finalize(); }

    // Non-copyable, non-movable
    scoped_shmem_environment(const scoped_shmem_environment&) = delete;
    scoped_shmem_environment& operator=(const scoped_shmem_environment&) = delete;
};

}  // namespace shmem
}  // namespace dtl
