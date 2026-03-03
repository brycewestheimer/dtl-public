// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file rma.cpp
 * @brief DTL Python bindings - RMA operations
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

#include <cstring>
#include <vector>

namespace py = pybind11;

// ============================================================================
// Window Wrapper Class
// ============================================================================

class PyWindow {
public:
    dtl_window_t win;
    bool owns;
    py::object base_owner;

    PyWindow() : win(nullptr), owns(false), base_owner(py::none()) {}

    PyWindow(dtl_window_t w, bool owner = true)
        : win(w), owns(owner), base_owner(py::none()) {}

    ~PyWindow() {
        if (owns && win) {
            dtl_window_destroy(win);
        }
    }

    // Prevent copying
    PyWindow(const PyWindow&) = delete;
    PyWindow& operator=(const PyWindow&) = delete;

    // Allow moving
    PyWindow(PyWindow&& other) noexcept
        : win(other.win), owns(other.owns), base_owner(std::move(other.base_owner)) {
        other.win = nullptr;
        other.owns = false;
        other.base_owner = py::none();
    }

    PyWindow& operator=(PyWindow&& other) noexcept {
        if (this != &other) {
            if (owns && win) {
                dtl_window_destroy(win);
            }
            win = other.win;
            owns = other.owns;
            base_owner = std::move(other.base_owner);
            other.win = nullptr;
            other.owns = false;
            other.base_owner = py::none();
        }
        return *this;
    }

    bool is_valid() const {
        return dtl_window_is_valid(win) != 0;
    }

    py::object base() const {
        void* ptr = dtl_window_base(win);
        if (!ptr) {
            return py::none();
        }
        // Return as integer (address) - users can interpret as needed
        return py::int_(reinterpret_cast<uintptr_t>(ptr));
    }

    size_t size() const {
        return dtl_window_size(win);
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

static void check_status(dtl_status status, const char* operation) {
    if (status != DTL_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + dtl_status_message(status));
    }
}

static dtl_dtype numpy_to_dtype(const py::dtype& dtype) {
    if (dtype.is(py::dtype::of<int8_t>())) return DTL_DTYPE_INT8;
    if (dtype.is(py::dtype::of<int16_t>())) return DTL_DTYPE_INT16;
    if (dtype.is(py::dtype::of<int32_t>())) return DTL_DTYPE_INT32;
    if (dtype.is(py::dtype::of<int64_t>())) return DTL_DTYPE_INT64;
    if (dtype.is(py::dtype::of<uint8_t>())) return DTL_DTYPE_UINT8;
    if (dtype.is(py::dtype::of<uint16_t>())) return DTL_DTYPE_UINT16;
    if (dtype.is(py::dtype::of<uint32_t>())) return DTL_DTYPE_UINT32;
    if (dtype.is(py::dtype::of<uint64_t>())) return DTL_DTYPE_UINT64;
    if (dtype.is(py::dtype::of<float>())) return DTL_DTYPE_FLOAT32;
    if (dtype.is(py::dtype::of<double>())) return DTL_DTYPE_FLOAT64;
    throw std::invalid_argument("Unsupported dtype");
}

static dtl_reduce_op string_to_reduce_op(const std::string& op) {
    if (op == "sum") return DTL_OP_SUM;
    if (op == "prod") return DTL_OP_PROD;
    if (op == "min") return DTL_OP_MIN;
    if (op == "max") return DTL_OP_MAX;
    if (op == "band") return DTL_OP_BAND;
    if (op == "bor") return DTL_OP_BOR;
    if (op == "bxor") return DTL_OP_BXOR;
    throw std::invalid_argument("Unknown reduction operation: " + op);
}

// Helper to create a 1D array from dtype and count
static py::array create_array(const py::dtype& dtype, py::ssize_t count) {
    std::vector<py::ssize_t> shape{count};
    return py::array(dtype, shape);
}

// Helper to convert Python object to bytes for a given dtype
static void python_to_bytes(const py::object& obj, const py::dtype& dtype, void* buffer) {
    // Create a single-element array and use numpy's conversion
    py::array arr = py::array(dtype, std::vector<py::ssize_t>{1});
    // Use numpy to do the conversion properly
    py::module_ np = py::module_::import("numpy");
    py::array converted = np.attr("asarray")(obj, dtype);
    std::memcpy(buffer, converted.data(), dtype.itemsize());
}

// Helper to convert bytes to Python object for a given dtype
static py::object bytes_to_python(const void* buffer, const py::dtype& dtype) {
    py::array arr = create_array(dtype, 1);
    std::memcpy(arr.mutable_data(), buffer, dtype.itemsize());
    return arr.attr("item")();
}

// ============================================================================
// Module Binding
// ============================================================================

// Helper to get context handle from either native Context or Python wrapper
static dtl_context_t get_context_handle(py::object ctx_obj) {
    // Check if this is a Python wrapper with _native property
    if (py::hasattr(ctx_obj, "_native")) {
        // Python wrapper: ctx._native.native()
        py::object native_ctx = ctx_obj.attr("_native");
        std::uintptr_t ctx_ptr = native_ctx.attr("native")().cast<std::uintptr_t>();
        return reinterpret_cast<dtl_context_t>(ctx_ptr);
    } else if (py::hasattr(ctx_obj, "native")) {
        // Native context directly: ctx.native()
        std::uintptr_t ctx_ptr = ctx_obj.attr("native")().cast<std::uintptr_t>();
        return reinterpret_cast<dtl_context_t>(ctx_ptr);
    } else {
        throw std::invalid_argument("Invalid context object");
    }
}

void bind_rma(py::module_& m) {
    // Window class
    py::class_<PyWindow>(m, "Window",
        "RMA memory window for one-sided communication")
        .def(py::init([](py::object ctx_obj, size_t size, py::object base_array) {
            // Get context handle
            dtl_context_t ctx = get_context_handle(ctx_obj);

            PyWindow win;
            dtl_status status;

            if (base_array.is_none()) {
                // Allocate window
                py::gil_scoped_release release;
                status = dtl_window_allocate(ctx, size, &win.win);
            } else {
                // Create window from numpy array
                py::array arr = base_array.cast<py::array>();
                void* ptr = arr.mutable_data();
                size_t arr_size = arr.nbytes();
                if (size == 0) {
                    size = arr_size;
                }
                // Keep base array alive for the full lifetime of the window.
                win.base_owner = base_array;
                py::gil_scoped_release release;
                status = dtl_window_create(ctx, ptr, size, &win.win);
            }

            check_status(status, "Window creation");
            win.owns = true;
            return win;
        }),
        py::arg("ctx"),
        py::arg("size") = 0,
        py::arg("base") = py::none(),
        "Create an RMA window. If base is None, allocates memory. "
        "Otherwise wraps the provided numpy array.")

        .def_property_readonly("base", &PyWindow::base,
            "Base pointer of the window (as integer address)")

        .def_property_readonly("size", &PyWindow::size,
            "Size of the window in bytes")

        .def_property_readonly("is_valid", &PyWindow::is_valid,
            "Check if the window handle is valid")

        // Fence synchronization
        .def("fence", [](PyWindow& self) {
            check_status(dtl_window_fence(self.win), "fence");
        },
        py::call_guard<py::gil_scoped_release>(),
        "Synchronize with a fence (collective)")

        // Lock/unlock
        .def("lock", [](PyWindow& self, int target, const std::string& mode) {
            dtl_lock_mode lock_mode = (mode == "exclusive") ?
                DTL_LOCK_EXCLUSIVE : DTL_LOCK_SHARED;
            check_status(dtl_window_lock(self.win, target, lock_mode), "lock");
        },
        py::arg("target"),
        py::arg("mode") = "exclusive",
        py::call_guard<py::gil_scoped_release>(),
        "Lock a target rank's window")

        .def("unlock", [](PyWindow& self, int target) {
            check_status(dtl_window_unlock(self.win, target), "unlock");
        },
        py::arg("target"),
        py::call_guard<py::gil_scoped_release>(),
        "Unlock a target rank's window")

        .def("lock_all", [](PyWindow& self) {
            check_status(dtl_window_lock_all(self.win), "lock_all");
        },
        py::call_guard<py::gil_scoped_release>(),
        "Lock all target ranks' windows")

        .def("unlock_all", [](PyWindow& self) {
            check_status(dtl_window_unlock_all(self.win), "unlock_all");
        },
        py::call_guard<py::gil_scoped_release>(),
        "Unlock all target ranks' windows")

        // Flush
        .def("flush", [](PyWindow& self, int target) {
            check_status(dtl_window_flush(self.win, target), "flush");
        },
        py::arg("target"),
        py::call_guard<py::gil_scoped_release>(),
        "Flush pending operations to a target")

        .def("flush_all", [](PyWindow& self) {
            check_status(dtl_window_flush_all(self.win), "flush_all");
        },
        py::call_guard<py::gil_scoped_release>(),
        "Flush pending operations to all targets")

        .def("flush_local", [](PyWindow& self, int target) {
            check_status(dtl_window_flush_local(self.win, target), "flush_local");
        },
        py::arg("target"),
        py::call_guard<py::gil_scoped_release>(),
        "Flush local completion for a target")

        .def("flush_local_all", [](PyWindow& self) {
            check_status(dtl_window_flush_local_all(self.win), "flush_local_all");
        },
        py::call_guard<py::gil_scoped_release>(),
        "Flush local completion for all targets")

        // Context managers for fence and lock epochs
        .def("__enter__", [](PyWindow& self) -> PyWindow& {
            return self;
        })

        .def("__exit__", [](PyWindow& self, py::object, py::object, py::object) {
            // Default: do nothing on exit
            // Users should explicitly call fence() or unlock()
        });

    // Free functions for RMA operations

    // Put
    m.def("rma_put", [](PyWindow& win, int target, size_t offset, py::array data) {
        const void* data_ptr = data.data();
        size_t nbytes = data.nbytes();
        py::gil_scoped_release release;
        check_status(dtl_rma_put(win.win, target, offset,
                                  data_ptr, nbytes),
                     "rma_put");
    },
    py::arg("window"),
    py::arg("target"),
    py::arg("offset"),
    py::arg("data"),
    "Put data into a remote window");

    // Get
    m.def("rma_get", [](PyWindow& win, int target, size_t offset,
                        size_t size, py::dtype dtype) -> py::array {
        // Calculate number of elements
        size_t elem_size = dtype.itemsize();
        size_t count = size / elem_size;

        // Create output array
        py::array result = create_array(dtype, static_cast<py::ssize_t>(count));

        void* result_ptr = result.mutable_data();
        {
            py::gil_scoped_release release;
            check_status(dtl_rma_get(win.win, target, offset,
                                      result_ptr, size),
                         "rma_get");
        }

        return result;
    },
    py::arg("window"),
    py::arg("target"),
    py::arg("offset"),
    py::arg("size"),
    py::arg("dtype"),
    "Get data from a remote window");

    // Accumulate
    m.def("rma_accumulate", [](PyWindow& win, int target, size_t offset,
                               py::array data, const std::string& op) {
        dtl_dtype dtype = numpy_to_dtype(data.dtype());
        dtl_reduce_op reduce_op = string_to_reduce_op(op);
        const void* data_ptr = data.data();
        size_t nbytes = data.nbytes();

        py::gil_scoped_release release;
        check_status(dtl_rma_accumulate(win.win, target, offset,
                                         data_ptr, nbytes,
                                         dtype, reduce_op),
                     "rma_accumulate");
    },
    py::arg("window"),
    py::arg("target"),
    py::arg("offset"),
    py::arg("data"),
    py::arg("op") = "sum",
    "Atomic accumulate operation on remote window");

    // Fetch and add (convenience wrapper)
    m.def("rma_fetch_and_add", [](PyWindow& win, int target, size_t offset,
                                  py::object addend, py::dtype dtype) -> py::object {
        dtl_dtype dt = numpy_to_dtype(dtype);
        size_t elem_size = dtl_dtype_size(dt);

        // Allocate buffers for origin and result
        std::vector<uint8_t> origin_buf(elem_size);
        std::vector<uint8_t> result_buf(elem_size);

        // Convert addend to bytes
        python_to_bytes(addend, dtype, origin_buf.data());

        {
            py::gil_scoped_release release;
            check_status(dtl_rma_fetch_and_op(win.win, target, offset,
                                               origin_buf.data(), result_buf.data(),
                                               dt, DTL_OP_SUM),
                         "rma_fetch_and_add");
        }

        // Convert result back to Python
        return bytes_to_python(result_buf.data(), dtype);
    },
    py::arg("window"),
    py::arg("target"),
    py::arg("offset"),
    py::arg("addend"),
    py::arg("dtype"),
    "Fetch old value and add to remote location");

    // Compare and swap
    m.def("rma_compare_and_swap", [](PyWindow& win, int target, size_t offset,
                                     py::object compare, py::object swap,
                                     py::dtype dtype) -> py::object {
        dtl_dtype dt = numpy_to_dtype(dtype);
        size_t elem_size = dtl_dtype_size(dt);

        // Allocate buffers
        std::vector<uint8_t> compare_buf(elem_size);
        std::vector<uint8_t> swap_buf(elem_size);
        std::vector<uint8_t> result_buf(elem_size);

        // Convert inputs to bytes
        python_to_bytes(compare, dtype, compare_buf.data());
        python_to_bytes(swap, dtype, swap_buf.data());

        {
            py::gil_scoped_release release;
            check_status(dtl_rma_compare_and_swap(win.win, target, offset,
                                                   compare_buf.data(), swap_buf.data(),
                                                   result_buf.data(), dt),
                         "rma_compare_and_swap");
        }

        // Convert result back to Python
        return bytes_to_python(result_buf.data(), dtype);
    },
    py::arg("window"),
    py::arg("target"),
    py::arg("offset"),
    py::arg("compare"),
    py::arg("swap"),
    py::arg("dtype"),
    "Atomic compare and swap on remote location");
}
