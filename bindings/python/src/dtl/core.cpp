// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file core.cpp
 * @brief DTL Python bindings - Core module (Context, exceptions)
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"

#include <memory>
#include <string>

#ifdef DTL_HAS_MPI4PY
#include <mpi4py/mpi4py.h>
#endif

namespace py = pybind11;

// ============================================================================
// Exception Handling
// ============================================================================

namespace {

/**
 * @brief Convert DTL status to Python exception
 */
void check_status(dtl_status status) {
    ::dtl::python::check_status_or_throw(status);
}

}  // namespace

// ============================================================================
// Context Wrapper Class
// ============================================================================

/**
 * @brief Python wrapper for dtl_context_t
 */
class DTLContext {
public:
    /**
     * @brief Create context with optional mpi4py communicator
     */
    DTLContext(py::object comm, int device_id) {
#ifdef DTL_HAS_MPI4PY
        if (!comm.is_none()) {
            // Import mpi4py and extract MPI_Comm
            if (import_mpi4py() < 0) {
                throw std::runtime_error("Failed to import mpi4py");
            }

            // Get MPI_Comm from mpi4py object
            // Note: This is a simplified version - full implementation would
            // use PyMPIComm_Get to extract the communicator
        }
#else
        if (!comm.is_none()) {
            throw std::runtime_error("mpi4py support not available");
        }
#endif

        // Create options
        dtl_context_options opts;
        dtl_context_options_init(&opts);
        opts.device_id = device_id;

        // Create context
        dtl_status status = dtl_context_create(&ctx_, &opts);
        check_status(status);
    }

    ~DTLContext() {
        if (ctx_) {
            dtl_context_destroy(ctx_);
        }
    }

    // Disable copy
    DTLContext(const DTLContext&) = delete;
    DTLContext& operator=(const DTLContext&) = delete;

    // Enable move
    DTLContext(DTLContext&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }

    DTLContext& operator=(DTLContext&& other) noexcept {
        if (this != &other) {
            if (ctx_) {
                dtl_context_destroy(ctx_);
            }
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    // =========================================================================
    // Basic Properties
    // =========================================================================

    dtl_rank_t rank() const { return dtl_context_rank(ctx_); }
    dtl_rank_t size() const { return dtl_context_size(ctx_); }
    bool is_root() const { return dtl_context_is_root(ctx_) != 0; }
    int device_id() const { return dtl_context_device_id(ctx_); }
    bool has_device() const { return dtl_context_has_device(ctx_) != 0; }

    // =========================================================================
    // Domain Query Properties (V1.3.0)
    // =========================================================================

    /// @brief Check if context has MPI domain
    bool has_mpi() const { return dtl_context_has_mpi(ctx_) != 0; }

    /// @brief Check if context has CUDA domain
    bool has_cuda() const { return dtl_context_has_cuda(ctx_) != 0; }

    /// @brief Check if context has NCCL domain
    bool has_nccl() const { return dtl_context_has_nccl(ctx_) != 0; }

    /// @brief Check if context has SHMEM domain
    bool has_shmem() const { return dtl_context_has_shmem(ctx_) != 0; }

    /// @brief Check if context handle is valid
    bool is_valid() const { return dtl_context_is_valid(ctx_) != 0; }

    // =========================================================================
    // Synchronization Methods
    // =========================================================================

    void barrier() {
        dtl_status status = dtl_context_barrier(ctx_);
        check_status(status);
    }

    void fence() {
        dtl_status status = dtl_context_fence(ctx_);
        check_status(status);
    }

    // =========================================================================
    // Context Factory Methods (V1.3.0)
    // =========================================================================

    /**
     * @brief Duplicate context with a new MPI communicator
     * @return New context with duplicated communicator
     */
    std::unique_ptr<DTLContext> dup() const {
        dtl_context_t new_ctx = nullptr;
        dtl_status status = dtl_context_dup(ctx_, &new_ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(new_ctx));
    }

    /**
     * @brief Split context by color (collective operation)
     * @param color Color for grouping (ranks with same color in same group)
     * @param key Ordering key within color group
     * @return New context with split communicator
     */
    std::unique_ptr<DTLContext> split(int color, int key = 0) const {
        dtl_context_t new_ctx = nullptr;
        dtl_status status = dtl_context_split(ctx_, color, key, &new_ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(new_ctx));
    }

    /**
     * @brief Create new context with CUDA domain added
     * @param device_id CUDA device ID
     * @return New context with CUDA domain
     */
    std::unique_ptr<DTLContext> with_cuda(int device_id) const {
        dtl_context_t new_ctx = nullptr;
        dtl_status status = dtl_context_with_cuda(ctx_, device_id, &new_ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(new_ctx));
    }

    /**
     * @brief Create new context with NCCL domain added
     * @param device_id CUDA device ID for NCCL
     * @return New context with NCCL domain
     * @note Requires MPI domain to be present
     */
    std::unique_ptr<DTLContext> with_nccl(int device_id) const {
        dtl_context_t new_ctx = nullptr;
        dtl_status status = dtl_context_with_nccl(ctx_, device_id, &new_ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(new_ctx));
    }

    /**
     * @brief Split context creating sub-groups with NCCL communicators
     * @param color Color for grouping (ranks with same color in same group)
     * @param key Ordering key within color group
     * @return New context with split MPI and NCCL communicators
     * @note Requires both MPI and NCCL domains
     */
    std::unique_ptr<DTLContext> split_nccl(int color, int key = 0) const {
        dtl_context_t new_ctx = nullptr;
        dtl_status status = dtl_context_split_nccl(ctx_, color, key, &new_ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(new_ctx));
    }

    // =========================================================================
    // Internal Access
    // =========================================================================

    /// @brief Get native context handle (for internal use)
    std::uintptr_t native() const { return reinterpret_cast<std::uintptr_t>(ctx_); }

private:
    dtl_context_t ctx_ = nullptr;

    friend class DTLEnvironment;

    /**
     * @brief Private constructor from existing context handle
     * @param ctx Existing context handle (takes ownership)
     */
    explicit DTLContext(dtl_context_t ctx) : ctx_(ctx) {}
};

// ============================================================================
// Environment Wrapper Class
// ============================================================================

/**
 * @brief Python wrapper for dtl_environment_t
 */
class DTLEnvironment {
public:
    DTLEnvironment() {
        dtl_status status = dtl_environment_create(&env_);
        check_status(status);
    }

    ~DTLEnvironment() {
        if (env_) {
            dtl_environment_destroy(env_);
        }
    }

    // Disable copy
    DTLEnvironment(const DTLEnvironment&) = delete;
    DTLEnvironment& operator=(const DTLEnvironment&) = delete;

    // Enable move
    DTLEnvironment(DTLEnvironment&& other) noexcept : env_(other.env_) {
        other.env_ = nullptr;
    }

    DTLEnvironment& operator=(DTLEnvironment&& other) noexcept {
        if (this != &other) {
            if (env_) {
                dtl_environment_destroy(env_);
            }
            env_ = other.env_;
            other.env_ = nullptr;
        }
        return *this;
    }

    // =========================================================================
    // State Queries (static — query the global registry)
    // =========================================================================

    static bool is_initialized() { return dtl_environment_is_initialized() != 0; }
    static dtl_size_t ref_count() { return dtl_environment_ref_count(); }

    // =========================================================================
    // Backend Availability (instance methods — delegate to registry)
    // =========================================================================

    bool has_mpi() const { return dtl_environment_has_mpi() != 0; }
    bool has_cuda() const { return dtl_environment_has_cuda() != 0; }
    bool has_hip() const { return dtl_environment_has_hip() != 0; }
    bool has_nccl() const { return dtl_environment_has_nccl() != 0; }
    bool has_shmem() const { return dtl_environment_has_shmem() != 0; }
    int mpi_thread_level() const { return dtl_environment_mpi_thread_level(); }

    // =========================================================================
    // Domain Query (instance, V1.4.0)
    // =========================================================================

    std::string domain() const {
        if (!env_) return "unknown";
        return dtl_environment_domain(env_);
    }

    // =========================================================================
    // Context Factories
    // =========================================================================

    std::unique_ptr<DTLContext> make_world_context() {
        dtl_context_t ctx = nullptr;
        dtl_status status = dtl_environment_make_world_context(env_, &ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(ctx));
    }

    std::unique_ptr<DTLContext> make_world_context_gpu(int device_id) {
        dtl_context_t ctx = nullptr;
        dtl_status status = dtl_environment_make_world_context_gpu(env_, device_id, &ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(ctx));
    }

    std::unique_ptr<DTLContext> make_cpu_context() {
        dtl_context_t ctx = nullptr;
        dtl_status status = dtl_environment_make_cpu_context(env_, &ctx);
        check_status(status);
        return std::unique_ptr<DTLContext>(new DTLContext(ctx));
    }

private:
    dtl_environment_t env_ = nullptr;
};

// ============================================================================
// Module Binding
// ============================================================================

void bind_core(py::module_& m) {
    // Context class
    py::class_<DTLContext>(m, "Context")
        .def(py::init<py::object, int>(),
             py::arg("comm") = py::none(),
             py::arg("device_id") = -1,
             R"doc(
Create a DTL context.

Args:
    comm: mpi4py MPI.Comm object (optional, None for MPI_COMM_WORLD)
    device_id: GPU device ID (default: -1 for CPU-only)

Example:
    >>> ctx = Context()
    >>> print(f"Rank {ctx.rank} of {ctx.size}")

    # Create GPU context
    >>> gpu_ctx = ctx.with_cuda(device_id=0)

    # Split context by even/odd ranks
    >>> even_ctx = ctx.split(color=ctx.rank % 2)
)doc")
        // Basic properties
        .def_property_readonly("rank", &DTLContext::rank,
                               "Current rank (0 to size-1)")
        .def_property_readonly("size", &DTLContext::size,
                               "Total number of ranks")
        .def_property_readonly("is_root", &DTLContext::is_root,
                               "True if this is rank 0")
        .def_property_readonly("device_id", &DTLContext::device_id,
                               "GPU device ID (-1 for CPU-only)")
        .def_property_readonly("has_device", &DTLContext::has_device,
                               "True if GPU is enabled")

        // Domain query properties (V1.3.0)
        .def_property_readonly("has_mpi", &DTLContext::has_mpi,
                               "True if MPI domain is available")
        .def_property_readonly("has_cuda", &DTLContext::has_cuda,
                               "True if CUDA domain is available")
        .def_property_readonly("has_nccl", &DTLContext::has_nccl,
                               "True if NCCL domain is available")
        .def_property_readonly("has_shmem", &DTLContext::has_shmem,
                               "True if SHMEM domain is available")
        .def_property_readonly("is_valid", &DTLContext::is_valid,
                               "True if context handle is valid")

        // Synchronization methods
        .def("barrier", &DTLContext::barrier,
             py::call_guard<py::gil_scoped_release>(),
             "Synchronize all ranks (collective)")
        .def("fence", &DTLContext::fence,
             py::call_guard<py::gil_scoped_release>(),
             "Memory fence (local synchronization)")

        // Context factory methods (V1.3.0)
        .def("dup", &DTLContext::dup,
             py::call_guard<py::gil_scoped_release>(),
             R"doc(
Duplicate context with a new MPI communicator.

Returns:
    New Context with duplicated communicator.

Note:
    This is a collective operation - all ranks must call it.
)doc")
        .def("split", &DTLContext::split,
             py::arg("color"),
             py::arg("key") = 0,
             py::call_guard<py::gil_scoped_release>(),
             R"doc(
Split context by color to create sub-groups.

Args:
    color: Color for grouping (ranks with same color form a group)
    key: Ordering key within color group (default: 0)

Returns:
    New Context with split communicator.

Note:
    This is a collective operation - all ranks must call it.

Example:
    # Split into even/odd groups
    >>> sub_ctx = ctx.split(color=ctx.rank % 2)

    # Split into groups of 4
    >>> sub_ctx = ctx.split(color=ctx.rank // 4)
)doc")
        .def("with_cuda", &DTLContext::with_cuda,
             py::arg("device_id"),
             R"doc(
Create new context with CUDA domain added.

Args:
    device_id: CUDA device ID to use

Returns:
    New Context with CUDA domain enabled.

Example:
    >>> gpu_ctx = ctx.with_cuda(device_id=0)
    >>> print(gpu_ctx.has_cuda)  # True
)doc")
        .def("with_nccl", &DTLContext::with_nccl,
             py::arg("device_id"),
             R"doc(
Create new context with NCCL domain added.

Args:
    device_id: CUDA device ID for NCCL communication

Returns:
    New Context with NCCL domain enabled.

Note:
    Requires MPI domain to be present (NCCL initialization
    uses MPI for rank coordination).

Example:
    >>> nccl_ctx = ctx.with_nccl(device_id=0)
    >>> print(nccl_ctx.has_nccl)  # True
)doc")
        .def("split_nccl", &DTLContext::split_nccl,
             py::arg("color"),
             py::arg("key") = 0,
             py::call_guard<py::gil_scoped_release>(),
             R"doc(
Split context creating sub-groups with NCCL communicators.

Args:
    color: Color for grouping (ranks with same color form a group)
    key: Ordering key within color group (default: 0)

Returns:
    New Context with split MPI and NCCL communicators.

Note:
    Requires both MPI and NCCL domains. This is a collective operation.
)doc")

        // Internal access
        .def("native", &DTLContext::native,
             "Get native context handle (for internal use)")

        // String representation
        .def("__repr__", [](const DTLContext& ctx) {
            std::string domains;
            if (ctx.has_mpi()) domains += "mpi,";
            if (ctx.has_cuda()) domains += "cuda,";
            if (ctx.has_nccl()) domains += "nccl,";
            if (ctx.has_shmem()) domains += "shmem,";
            if (!domains.empty()) domains.pop_back();  // Remove trailing comma

            return "<Context rank=" + std::to_string(ctx.rank()) +
                   " size=" + std::to_string(ctx.size()) +
                   " device=" + std::to_string(ctx.device_id()) +
                   " domains=[" + domains + "]>";
        });

    // Environment class
    py::class_<DTLEnvironment>(m, "Environment")
        .def(py::init<>(),
             R"doc(
Create a DTL environment.

Initializes all backends (MPI, CUDA, etc.) on first creation.
Subsequent creations increment a reference count. The last
destruction finalizes backends.

Example:
    >>> env = dtl.Environment()
    >>> ctx = env.make_world_context()
    >>> print(f"Rank {ctx.rank} of {ctx.size}")

    # Also works as context manager:
    >>> with dtl.Environment() as env:
    ...     ctx = env.make_world_context()
)doc")
        // Static queries
        .def_static("is_initialized", &DTLEnvironment::is_initialized,
                     "True if at least one environment handle exists")
        .def_static("ref_count", &DTLEnvironment::ref_count,
                     "Number of active environment handles")

        // Backend availability (instance properties)
        .def_property_readonly("has_mpi", &DTLEnvironment::has_mpi,
            "True if MPI backend is available")
        .def_property_readonly("has_cuda", &DTLEnvironment::has_cuda,
            "True if CUDA backend is available")
        .def_property_readonly("has_hip", &DTLEnvironment::has_hip,
            "True if HIP backend is available")
        .def_property_readonly("has_nccl", &DTLEnvironment::has_nccl,
            "True if NCCL backend is available")
        .def_property_readonly("has_shmem", &DTLEnvironment::has_shmem,
            "True if SHMEM backend is available")
        .def("mpi_thread_level", &DTLEnvironment::mpi_thread_level,
             "MPI thread level (0-3), or -1 if not available")

        // Domain query (V1.4.0)
        .def_property_readonly("domain", &DTLEnvironment::domain,
            "Named domain label for this environment")

        // Context factories
        .def("make_world_context", &DTLEnvironment::make_world_context,
             R"doc(
Create a world context spanning all MPI ranks.

Returns:
    Context with MPI and CPU domains.

Example:
    >>> ctx = env.make_world_context()
    >>> print(f"Rank {ctx.rank} of {ctx.size}")
)doc")
        .def("make_world_context", &DTLEnvironment::make_world_context_gpu,
             py::arg("device_id"),
             R"doc(
Create a GPU-enabled world context.

Args:
    device_id: CUDA device ID to use.

Returns:
    Context with MPI, CPU, and CUDA domains.

Example:
    >>> ctx = env.make_world_context(device_id=0)
    >>> print(ctx.has_cuda)  # True
)doc")
        .def("make_cpu_context", &DTLEnvironment::make_cpu_context,
             R"doc(
Create a CPU-only context (single-process, no MPI).

Returns:
    Context with CPU domain only.

Example:
    >>> ctx = env.make_cpu_context()
    >>> print(ctx.rank)  # 0
)doc")

        // String representation
        .def("__repr__", [](const DTLEnvironment& env) {
            std::string backends;
            if (env.has_mpi()) backends += "mpi,";
            if (env.has_cuda()) backends += "cuda,";
            if (env.has_hip()) backends += "hip,";
            if (env.has_nccl()) backends += "nccl,";
            if (env.has_shmem()) backends += "shmem,";
            if (!backends.empty()) backends.pop_back();
            return "<Environment domain=" + env.domain() +
                   " backends=[" + backends + "] refcount=" +
                   std::to_string(DTLEnvironment::ref_count()) + ">";
        });
}
