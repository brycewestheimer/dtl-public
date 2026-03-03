// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file mpi_request.hpp
/// @brief RAII wrapper for MPI_Request handles
/// @details Provides automatic cleanup of non-blocking MPI request handles.
///          When an mpi_request is destroyed with a still-pending operation,
///          the destructor cancels the request and waits for completion to
///          prevent resource leaks.
///
/// @warning Callers MUST either wait on or cancel MPI requests before the
///          underlying communication buffers go out of scope. While this RAII
///          wrapper prevents MPI_Request handle leaks, it cannot prevent
///          use-after-free of user buffers passed to MPI_Isend/MPI_Irecv.
///          The destructor will cancel abandoned requests, but this is a
///          safety net -- not a substitute for proper request lifecycle
///          management.
///
/// @since 0.1.0

#pragma once

#include <dtl/core/config.hpp>
#include <dtl/core/types.hpp>
#include <dtl/backend/concepts/communicator.hpp>

#if DTL_ENABLE_MPI
#include <mpi.h>
#endif

#include <utility>

namespace dtl {
namespace mpi {

// ============================================================================
// MPI Request RAII Wrapper
// ============================================================================

/// @brief RAII wrapper for a heap-allocated MPI_Request handle
///
/// @details Owns a heap-allocated `MPI_Request*` as returned by
///          `mpi_communicator::isend_impl()` / `irecv_impl()` and friends.
///          Ensures that abandoned (non-completed) requests are cancelled and
///          cleaned up in the destructor, preventing MPI resource leaks.
///
///          This class is move-only (non-copyable) to maintain unique
///          ownership of the underlying MPI_Request.
///
/// @par Usage example:
/// @code
///   auto handle = comm.isend_impl(buf, count, 1, dest, tag);
///   if (handle) {
///       dtl::mpi::mpi_request req(handle.value());
///       // ... do other work ...
///       req.wait();  // blocks until send completes
///   }
/// @endcode
///
/// @par Constructing from request_handle:
/// @code
///   request_handle raw = ...;  // from isend_impl / irecv_impl
///   dtl::mpi::mpi_request req(raw);
///   // req now owns the MPI_Request; raw.handle is set to nullptr
/// @endcode
///
/// @warning You MUST ensure that the communication buffer passed to
///          MPI_Isend/MPI_Irecv remains valid until the request completes.
///          This wrapper only manages the MPI_Request handle lifetime,
///          not the buffer lifetime.
class mpi_request {
public:
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    /// @brief Native pointer type for the underlying MPI_Request
    /// @details When MPI is enabled, this is `MPI_Request*`.
    ///          When MPI is disabled, this is `void*` (stub).
#if DTL_ENABLE_MPI
    using native_handle_type = MPI_Request*;
#else
    using native_handle_type = void*;
#endif

    // -------------------------------------------------------------------------
    // Constructors / Destructor
    // -------------------------------------------------------------------------

    /// @brief Default constructor (creates an empty/completed request)
    mpi_request() noexcept = default;

    /// @brief Construct from a request_handle, taking ownership
    /// @param handle The request handle from isend_impl/irecv_impl.
    ///        On return, handle.handle is set to nullptr (ownership transferred).
    /// @details The mpi_request takes ownership of the heap-allocated
    ///          MPI_Request pointed to by handle.handle.
    explicit mpi_request(request_handle& handle) noexcept
        : req_(static_cast<native_handle_type>(handle.handle)) {
        handle.handle = nullptr;
    }

    /// @brief Construct directly from a raw MPI_Request pointer, taking ownership
    /// @param raw_req Pointer to a heap-allocated MPI_Request. Ownership is
    ///        transferred to this mpi_request. Pass nullptr for an empty request.
    explicit mpi_request(native_handle_type raw_req) noexcept
        : req_(raw_req) {}

    /// @brief Destructor -- cancels and cleans up any pending request
    /// @details If the owned MPI_Request is still pending (not MPI_REQUEST_NULL
    ///          and not already completed), calls MPI_Cancel followed by
    ///          MPI_Wait to ensure proper cleanup, then deletes the
    ///          heap-allocated MPI_Request.
    ~mpi_request() {
        cleanup();
    }

    // Non-copyable
    mpi_request(const mpi_request&) = delete;
    mpi_request& operator=(const mpi_request&) = delete;

    // -------------------------------------------------------------------------
    // Move Operations
    // -------------------------------------------------------------------------

    /// @brief Move constructor -- transfers ownership from other
    mpi_request(mpi_request&& other) noexcept
        : req_(other.req_) {
        other.req_ = nullptr;
    }

    /// @brief Move assignment -- transfers ownership from other
    mpi_request& operator=(mpi_request&& other) noexcept {
        if (this != &other) {
            cleanup();
            req_ = other.req_;
            other.req_ = nullptr;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Request Operations
    // -------------------------------------------------------------------------

    /// @brief Wait for the non-blocking operation to complete (blocking)
    /// @details Calls MPI_Wait on the underlying request. After successful
    ///          completion, the MPI_Request is set to MPI_REQUEST_NULL by MPI
    ///          and the heap allocation is freed.
    /// @return true if the wait succeeded, false on error or if no request
    ///         is owned
    bool wait() {
#if DTL_ENABLE_MPI
        if (!req_) {
            return true;  // No request to wait on -- considered "complete"
        }
        MPI_Status status;
        int result = MPI_Wait(req_, &status);
        // MPI_Wait sets *req_ to MPI_REQUEST_NULL on success
        delete req_;
        req_ = nullptr;
        return result == MPI_SUCCESS;
#else
        return true;
#endif
    }

    /// @brief Test if the non-blocking operation has completed (non-blocking)
    /// @details Calls MPI_Test on the underlying request. If the operation
    ///          has completed, the MPI_Request is set to MPI_REQUEST_NULL by
    ///          MPI and the heap allocation is freed.
    /// @return true if the operation is complete (or no request is owned),
    ///         false if still pending
    bool test() {
#if DTL_ENABLE_MPI
        if (!req_) {
            return true;  // No request -- considered "complete"
        }
        int flag = 0;
        MPI_Status status;
        int result = MPI_Test(req_, &flag, &status);
        if (result != MPI_SUCCESS) {
            return false;
        }
        if (flag) {
            // Request completed; MPI has set *req_ to MPI_REQUEST_NULL
            delete req_;
            req_ = nullptr;
        }
        return flag != 0;
#else
        return true;
#endif
    }

    /// @brief Cancel the pending request
    /// @details Marks the request for cancellation via MPI_Cancel, then waits
    ///          for the cancellation to complete via MPI_Wait. After this call,
    ///          the request is no longer owned.
    /// @return true if the cancellation succeeded, false on error or if no
    ///         request is owned
    bool cancel() {
#if DTL_ENABLE_MPI
        if (!req_) {
            return true;  // Nothing to cancel
        }
        if (*req_ == MPI_REQUEST_NULL) {
            delete req_;
            req_ = nullptr;
            return true;
        }
        MPI_Cancel(req_);
        MPI_Status status;
        MPI_Wait(req_, &status);
        delete req_;
        req_ = nullptr;
        return true;
#else
        return true;
#endif
    }

    /// @brief Release ownership of the raw MPI_Request pointer
    /// @details Returns the internal pointer and relinquishes ownership.
    ///          The caller becomes responsible for completing and deleting
    ///          the MPI_Request. This is provided for backward compatibility
    ///          with code that manages MPI_Request handles manually.
    /// @return The raw MPI_Request pointer (may be nullptr if no request owned)
    [[nodiscard]] native_handle_type release() noexcept {
        auto* ptr = req_;
        req_ = nullptr;
        return ptr;
    }

    /// @brief Convert back to a request_handle (releases ownership)
    /// @details Creates a request_handle from the owned pointer and releases
    ///          ownership. The returned handle must be managed by the caller
    ///          (e.g., passed to mpi_communicator::wait_impl).
    /// @return A request_handle owning the MPI_Request, or an invalid handle
    ///         if no request is owned
    [[nodiscard]] request_handle release_to_handle() noexcept {
        request_handle h;
        h.handle = static_cast<void*>(req_);
        req_ = nullptr;
        return h;
    }

    // -------------------------------------------------------------------------
    // Queries
    // -------------------------------------------------------------------------

    /// @brief Check if this wrapper owns a request
    /// @return true if an MPI_Request is owned (may or may not be pending)
    [[nodiscard]] bool valid() const noexcept {
        return req_ != nullptr;
    }

    /// @brief Boolean conversion -- true if a request is owned
    [[nodiscard]] explicit operator bool() const noexcept {
        return valid();
    }

private:
    // -------------------------------------------------------------------------
    // Internal Helpers
    // -------------------------------------------------------------------------

    /// @brief Clean up owned request (cancel if pending, then delete)
    void cleanup() noexcept {
#if DTL_ENABLE_MPI
        if (!req_) {
            return;
        }
        // If the request is still active (not MPI_REQUEST_NULL), cancel it
        if (*req_ != MPI_REQUEST_NULL) {
            // Check if MPI is still initialized -- calling MPI functions
            // after MPI_Finalize is undefined behavior
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Cancel(req_);
                MPI_Status status;
                MPI_Wait(req_, &status);
            }
        }
        delete req_;
        req_ = nullptr;
#endif
    }

    // -------------------------------------------------------------------------
    // Data Members
    // -------------------------------------------------------------------------

    /// @brief Owned heap-allocated MPI_Request (nullptr if empty/completed)
    native_handle_type req_ = nullptr;
};

}  // namespace mpi
}  // namespace dtl
