// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file mpmd.cpp
 * @brief DTL Python bindings - MPMD role manager operations
 * @since 0.1.0
 *
 * Provides Python bindings for Multiple Program Multiple Data (MPMD)
 * support, including role management and inter-group communication.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>
#include <dtl/bindings/c/dtl_mpmd.h>

#include "status_exception.hpp"

#include <stdexcept>
#include <string>

namespace py = pybind11;

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Convert DTL status to Python exception
 */
void check_status(dtl_status status) {
    ::dtl::python::check_status_or_throw(status);
}

/**
 * @brief Extract native context from Python Context object
 */
dtl_context_t get_native_context(py::object ctx_obj) {
    if (py::hasattr(ctx_obj, "native")) {
        py::object native_method = ctx_obj.attr("native");
        return reinterpret_cast<dtl_context_t>(
            native_method().cast<std::uintptr_t>());
    } else if (py::hasattr(ctx_obj, "_native")) {
        py::object inner_ctx = ctx_obj.attr("_native");
        if (py::hasattr(inner_ctx, "native")) {
            py::object native_method = inner_ctx.attr("native");
            return reinterpret_cast<dtl_context_t>(
                native_method().cast<std::uintptr_t>());
        }
    }
    throw std::runtime_error("Invalid context object - could not extract native handle");
}

/**
 * @brief Convert NumPy dtype to DTL dtype
 */
dtl_dtype numpy_to_dtl_dtype(py::dtype dtype) {
    if (dtype.equal(py::dtype::of<double>())) return DTL_DTYPE_FLOAT64;
    if (dtype.equal(py::dtype::of<float>())) return DTL_DTYPE_FLOAT32;
    if (dtype.equal(py::dtype::of<int64_t>())) return DTL_DTYPE_INT64;
    if (dtype.equal(py::dtype::of<int32_t>())) return DTL_DTYPE_INT32;
    if (dtype.equal(py::dtype::of<uint64_t>())) return DTL_DTYPE_UINT64;
    if (dtype.equal(py::dtype::of<uint32_t>())) return DTL_DTYPE_UINT32;
    if (dtype.equal(py::dtype::of<int16_t>())) return DTL_DTYPE_INT16;
    if (dtype.equal(py::dtype::of<uint16_t>())) return DTL_DTYPE_UINT16;
    if (dtype.equal(py::dtype::of<int8_t>())) return DTL_DTYPE_INT8;
    if (dtype.equal(py::dtype::of<uint8_t>())) return DTL_DTYPE_UINT8;
    throw std::runtime_error("Unsupported dtype for MPMD operation");
}

}  // namespace

// ============================================================================
// RoleManager Wrapper Class
// ============================================================================

/**
 * @brief Python wrapper for dtl_role_manager_t
 *
 * Manages MPMD role assignment and inter-group communication.
 */
class PyRoleManager {
public:
    /**
     * @brief Create a role manager bound to a context
     */
    explicit PyRoleManager(py::object ctx_obj) {
        dtl_context_t ctx = get_native_context(ctx_obj);
        dtl_status status = dtl_role_manager_create(ctx, &mgr_);
        check_status(status);
    }

    ~PyRoleManager() {
        if (mgr_) {
            dtl_role_manager_destroy(mgr_);
        }
    }

    // Disable copy
    PyRoleManager(const PyRoleManager&) = delete;
    PyRoleManager& operator=(const PyRoleManager&) = delete;

    // Enable move
    PyRoleManager(PyRoleManager&& other) noexcept : mgr_(other.mgr_) {
        other.mgr_ = nullptr;
    }

    PyRoleManager& operator=(PyRoleManager&& other) noexcept {
        if (this != &other) {
            if (mgr_) {
                dtl_role_manager_destroy(mgr_);
            }
            mgr_ = other.mgr_;
            other.mgr_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Add a named role with a specified number of ranks
     */
    void add_role(const std::string& name, dtl_size_t num_ranks) {
        if (!mgr_) {
            throw std::runtime_error("RoleManager has been destroyed");
        }
        dtl_status status = dtl_role_manager_add_role(mgr_, name.c_str(), num_ranks);
        check_status(status);
    }

    /**
     * @brief Initialize the role manager (assigns ranks to roles)
     */
    void initialize() {
        if (!mgr_) {
            throw std::runtime_error("RoleManager has been destroyed");
        }
        dtl_status status = dtl_role_manager_initialize(mgr_);
        check_status(status);
    }

    /**
     * @brief Check if this rank has the specified role
     */
    bool has_role(const std::string& name) {
        if (!mgr_) {
            throw std::runtime_error("RoleManager has been destroyed");
        }
        int result = 0;
        dtl_status status = dtl_role_manager_has_role(mgr_, name.c_str(), &result);
        check_status(status);
        return result != 0;
    }

    /**
     * @brief Get the number of ranks in a role
     */
    dtl_size_t role_size(const std::string& name) {
        if (!mgr_) {
            throw std::runtime_error("RoleManager has been destroyed");
        }
        dtl_size_t size = 0;
        dtl_status status = dtl_role_manager_role_size(mgr_, name.c_str(), &size);
        check_status(status);
        return size;
    }

    /**
     * @brief Get this rank's rank within a role group
     */
    dtl_rank_t role_rank(const std::string& name) {
        if (!mgr_) {
            throw std::runtime_error("RoleManager has been destroyed");
        }
        dtl_rank_t rank = 0;
        dtl_status status = dtl_role_manager_role_rank(mgr_, name.c_str(), &rank);
        check_status(status);
        return rank;
    }

    /**
     * @brief Explicitly destroy the role manager
     */
    void destroy() {
        if (mgr_) {
            dtl_role_manager_destroy(mgr_);
            mgr_ = nullptr;
        }
    }

    /**
     * @brief Get the native handle (for internal use)
     */
    std::uintptr_t native() const {
        return reinterpret_cast<std::uintptr_t>(mgr_);
    }

private:
    dtl_role_manager_t mgr_ = nullptr;
};

// ============================================================================
// Module Binding
// ============================================================================

void init_mpmd(py::module_& m) {
    auto mpmd = m.def_submodule("mpmd", "MPMD role manager and inter-group communication");

    // RoleManager class
    py::class_<PyRoleManager>(mpmd, "RoleManager",
        R"doc(
MPMD role manager for partitioning ranks into named groups.

Roles partition MPI ranks into named groups, enabling distinct programs
or phases to communicate across group boundaries.

Example:
    >>> mgr = dtl.mpmd.RoleManager(ctx)
    >>> mgr.add_role("producer", 2)
    >>> mgr.add_role("consumer", 2)
    >>> mgr.initialize()
    >>> if mgr.has_role("producer"):
    ...     print(f"Producer rank {mgr.role_rank('producer')}")
)doc")
        .def(py::init<py::object>(),
             py::arg("ctx"),
             "Create a role manager bound to a context")
        .def("add_role", &PyRoleManager::add_role,
             py::arg("name"),
             py::arg("num_ranks"),
             R"doc(
Add a named role with a specified number of ranks.

Must be called before initialize(). The total ranks across all roles
must equal the communicator size.

Args:
    name: Name of the role (must be unique)
    num_ranks: Number of ranks to assign to this role
)doc")
        .def("initialize", &PyRoleManager::initialize,
             R"doc(
Initialize the role manager.

Assigns ranks to roles sequentially in the order roles were added.
This is a collective operation - all ranks must call it.
)doc")
        .def("has_role", &PyRoleManager::has_role,
             py::arg("name"),
             R"doc(
Check if this rank has the specified role.

Args:
    name: Name of the role to check

Returns:
    True if this rank belongs to the named role
)doc")
        .def("role_size", &PyRoleManager::role_size,
             py::arg("name"),
             R"doc(
Get the number of ranks in a role.

Args:
    name: Name of the role to query

Returns:
    Number of ranks assigned to the role
)doc")
        .def("role_rank", &PyRoleManager::role_rank,
             py::arg("name"),
             R"doc(
Get this rank's local rank within a role group.

Args:
    name: Name of the role to query

Returns:
    Local rank index within the role group (0-based)
)doc")
        .def("destroy", &PyRoleManager::destroy,
             "Explicitly destroy the role manager and release resources")
        .def("native", &PyRoleManager::native,
             "Get native handle (for internal use)")
        .def("__repr__", [](const PyRoleManager& mgr) {
            return "<RoleManager handle=" +
                   std::to_string(mgr.native()) + ">";
        });

    // Inter-group communication functions

    mpmd.def("intergroup_send",
        [](PyRoleManager& mgr, const std::string& target_role,
           dtl_rank_t target_rank, py::array buf, int tag) {
            dtl_role_manager_t native_mgr =
                reinterpret_cast<dtl_role_manager_t>(mgr.native());
            if (!native_mgr) {
                throw std::runtime_error("RoleManager has been destroyed");
            }

            dtl_dtype dtype = numpy_to_dtl_dtype(buf.dtype());

            py::gil_scoped_release release;
            dtl_status status = dtl_intergroup_send(
                native_mgr,
                target_role.c_str(),
                target_rank,
                buf.data(),
                static_cast<dtl_size_t>(buf.size()),
                dtype,
                static_cast<dtl_tag_t>(tag)
            );
            check_status(status);
        },
        py::arg("mgr"),
        py::arg("target_role"),
        py::arg("target_rank"),
        py::arg("buf"),
        py::arg("tag") = 0,
        R"doc(
Send data to a rank in another role group.

Args:
    mgr: RoleManager instance
    target_role: Name of the target role group
    target_rank: Rank within the target role group (0 to role_size-1)
    buf: NumPy array to send
    tag: Message tag (default: 0)

Example:
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> dtl.mpmd.intergroup_send(mgr, "consumer", 0, data, tag=42)
)doc");

    mpmd.def("intergroup_recv",
        [](PyRoleManager& mgr, const std::string& source_role,
           dtl_rank_t source_rank, dtl_size_t count,
           py::dtype dtype, int tag) -> py::array {
            dtl_role_manager_t native_mgr =
                reinterpret_cast<dtl_role_manager_t>(mgr.native());
            if (!native_mgr) {
                throw std::runtime_error("RoleManager has been destroyed");
            }

            dtl_dtype dt = numpy_to_dtl_dtype(dtype);

            // Create output array with explicit shape container to avoid
            // ambiguous py::array overload resolution.
            py::array result(dtype,
                             py::array::ShapeContainer{static_cast<py::ssize_t>(count)});

            {
                py::gil_scoped_release release;
                dtl_status status = dtl_intergroup_recv(
                    native_mgr,
                    source_role.c_str(),
                    source_rank,
                    result.mutable_data(),
                    count,
                    dt,
                    static_cast<dtl_tag_t>(tag)
                );
                check_status(status);
            }

            return result;
        },
        py::arg("mgr"),
        py::arg("source_role"),
        py::arg("source_rank"),
        py::arg("count"),
        py::arg("dtype"),
        py::arg("tag") = 0,
        R"doc(
Receive data from a rank in another role group.

Args:
    mgr: RoleManager instance
    source_role: Name of the source role group
    source_rank: Rank within the source role group (0 to role_size-1)
    count: Number of elements to receive
    dtype: NumPy dtype of elements
    tag: Message tag (default: 0)

Returns:
    NumPy array containing the received data

Example:
    >>> data = dtl.mpmd.intergroup_recv(mgr, "producer", 0, 100,
    ...                                  np.float64, tag=42)
)doc");
}
