// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file connection_pool.cpp
/// @brief Communicator connection pooling implementations
/// @details MPI pool uses MPI_Comm_dup/MPI_Comm_free for isolated communicators.
///          NCCL returns not_supported (NCCL communicators are heavyweight and
///          should be reused directly, not pooled).
/// @since 0.1.0

#include <dtl/runtime/connection_pool.hpp>
#include <dtl/error/result.hpp>
#include <dtl/error/status.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if DTL_ENABLE_MPI
#  include <mpi.h>
#endif

namespace dtl::runtime {

namespace {

#if DTL_ENABLE_MPI

bool mpi_ready() {
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        return false;
    }

    int finalized = 0;
    MPI_Finalized(&finalized);
    return finalized == 0;
}

void free_comm(MPI_Comm comm) {
    if (!mpi_ready()) {
        return;
    }
    MPI_Comm_free(&comm);
}

struct mpi_pool_state {
    mutable std::mutex mtx;
    std::vector<MPI_Comm> idle;
    uint32_t capacity{8};
    pool_metrics metrics{};
    bool shutdown{false};
};

void release_comm_to_state(MPI_Comm comm, const std::shared_ptr<mpi_pool_state>& state) {
    bool should_free = false;
    {
        std::lock_guard lock(state->mtx);
        state->metrics.total_released++;
        if (state->metrics.current_active > 0) {
            state->metrics.current_active--;
        }

        if (!state->shutdown && state->idle.size() < state->capacity) {
            state->idle.push_back(comm);
            state->metrics.pool_size = static_cast<uint32_t>(state->idle.size());
            state->metrics.pool_capacity = state->capacity;
            return;
        }

        state->metrics.pool_size = static_cast<uint32_t>(state->idle.size());
        state->metrics.pool_capacity = state->capacity;
        should_free = true;
    }

    if (should_free) {
        free_comm(comm);
    }
}

#endif

}  // namespace

// =============================================================================
// MPI Communicator Pool
// =============================================================================

#if DTL_ENABLE_MPI

class mpi_communicator_pool final : public communicator_pool {
public:
    explicit mpi_communicator_pool(uint32_t initial_capacity = 8)
        : state_(std::make_shared<mpi_pool_state>()) {
        state_->capacity = initial_capacity;
        state_->metrics.pool_capacity = initial_capacity;
    }

    ~mpi_communicator_pool() override {
        std::vector<MPI_Comm> idle_to_free;
        {
            std::lock_guard lock(state_->mtx);
            state_->shutdown = true;
            idle_to_free.swap(state_->idle);
            state_->metrics.pool_size = 0;
        }
        for (auto comm : idle_to_free) {
            free_comm(comm);
        }
    }

    dtl::result<pool_handle> acquire() override {
        if (!mpi_ready()) {
            return dtl::make_error<pool_handle>(
                dtl::status_code::invalid_state,
                "MPI communicator pooling requires MPI to be initialized");
        }

        MPI_Comm comm;
        {
            std::lock_guard lock(state_->mtx);
            if (state_->shutdown) {
                return dtl::make_error<pool_handle>(
                    dtl::status_code::invalid_state,
                    "MPI communicator pool is shutting down");
            }

            if (!state_->idle.empty()) {
                comm = state_->idle.back();
                state_->idle.pop_back();
            } else {
                int rc = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
                if (rc != MPI_SUCCESS) {
                    return dtl::make_error<pool_handle>(
                        dtl::status_code::communication_error,
                        "MPI_Comm_dup failed with code " + std::to_string(rc));
                }
            }

            state_->metrics.total_acquired++;
            state_->metrics.current_active++;
            if (state_->metrics.current_active > state_->metrics.high_water_mark) {
                state_->metrics.high_water_mark = state_->metrics.current_active;
            }
            state_->metrics.pool_size = static_cast<uint32_t>(state_->idle.size());
            state_->metrics.pool_capacity = state_->capacity;
        }

        auto* heap_comm = new MPI_Comm(comm);
        auto state = state_;

        return pool_handle{
            static_cast<void*>(heap_comm),
            [state](void* resource) {
                auto* comm_ptr = static_cast<MPI_Comm*>(resource);
                release_comm_to_state(*comm_ptr, state);
                delete comm_ptr;
            }
        };
    }

    pool_metrics metrics() const noexcept override {
        std::lock_guard lock(state_->mtx);
        auto m = state_->metrics;
        m.pool_size = static_cast<uint32_t>(state_->idle.size());
        m.pool_capacity = state_->capacity;
        return m;
    }

    std::string_view backend_name() const noexcept override {
        return "mpi";
    }

    void set_capacity(uint32_t capacity) override {
        std::vector<MPI_Comm> to_free;
        {
            std::lock_guard lock(state_->mtx);
            state_->capacity = capacity;
            state_->metrics.pool_capacity = capacity;
            while (state_->idle.size() > state_->capacity) {
                to_free.push_back(state_->idle.back());
                state_->idle.pop_back();
            }
            state_->metrics.pool_size = static_cast<uint32_t>(state_->idle.size());
        }
        for (auto comm : to_free) {
            free_comm(comm);
        }
    }

    void drain() override {
        std::vector<MPI_Comm> to_free;
        {
            std::lock_guard lock(state_->mtx);
            to_free.swap(state_->idle);
            state_->metrics.pool_size = 0;
        }
        for (auto comm : to_free) {
            free_comm(comm);
        }
    }

private:
    std::shared_ptr<mpi_pool_state> state_;
};

#endif  // DTL_ENABLE_MPI

// =============================================================================
// Factory
// =============================================================================

dtl::result<std::unique_ptr<communicator_pool>>
make_communicator_pool(std::string_view backend) {
#if DTL_ENABLE_MPI
    if (backend == "mpi") {
        if (!mpi_ready()) {
            return dtl::make_error<std::unique_ptr<communicator_pool>>(
                dtl::status_code::invalid_state,
                "MPI communicator pooling requires MPI to be initialized");
        }
        return std::unique_ptr<communicator_pool>{
            std::make_unique<mpi_communicator_pool>()};
    }
#endif

    if (backend == "nccl") {
        return dtl::make_error<std::unique_ptr<communicator_pool>>(
            dtl::status_code::not_supported,
            "NCCL communicators are heavyweight and should be reused directly; "
            "pooling is not supported. Create ncclComm_t instances explicitly "
            "and manage their lifetime manually.");
    }

    return dtl::make_error<std::unique_ptr<communicator_pool>>(
        dtl::status_code::not_supported,
        "communicator pooling for backend '" + std::string(backend) +
        "' is not supported");
}

}  // namespace dtl::runtime
