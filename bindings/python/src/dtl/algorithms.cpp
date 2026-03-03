// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file algorithms.cpp
 * @brief DTL Python bindings - Algorithm operations on containers
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

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
 * @brief Extract native vector handle from Python DistributedVector object
 */
dtl_vector_t get_native_vector(py::object vec_obj) {
    if (py::hasattr(vec_obj, "_native")) {
        py::object inner = vec_obj.attr("_native");
        if (py::hasattr(inner, "native")) {
            py::object native_method = inner.attr("native");
            return reinterpret_cast<dtl_vector_t>(
                native_method().cast<std::uintptr_t>());
        }
    }
    if (py::hasattr(vec_obj, "native")) {
        py::object native_method = vec_obj.attr("native");
        return reinterpret_cast<dtl_vector_t>(
            native_method().cast<std::uintptr_t>());
    }
    throw std::runtime_error("Invalid vector object - could not extract native handle");
}

/**
 * @brief Extract native array handle from Python DistributedArray object
 */
dtl_array_t get_native_array(py::object arr_obj) {
    if (py::hasattr(arr_obj, "_native")) {
        py::object inner = arr_obj.attr("_native");
        if (py::hasattr(inner, "native")) {
            py::object native_method = inner.attr("native");
            return reinterpret_cast<dtl_array_t>(
                native_method().cast<std::uintptr_t>());
        }
    }
    if (py::hasattr(arr_obj, "native")) {
        py::object native_method = arr_obj.attr("native");
        return reinterpret_cast<dtl_array_t>(
            native_method().cast<std::uintptr_t>());
    }
    throw std::runtime_error("Invalid array object - could not extract native handle");
}

/**
 * @brief Convert string to DTL reduce operation
 */
dtl_reduce_op string_to_reduce_op(const std::string& op) {
    if (op == "sum" || op == "SUM") return DTL_OP_SUM;
    if (op == "prod" || op == "PROD") return DTL_OP_PROD;
    if (op == "min" || op == "MIN") return DTL_OP_MIN;
    if (op == "max" || op == "MAX") return DTL_OP_MAX;
    throw std::runtime_error("Unknown reduce operation: " + op);
}

// Callback context for Python function wrapping
struct PythonCallbackContext {
    py::function func;
    dtl_dtype dtype;
    bool with_index;
};

// For-each callback that invokes Python function
void python_for_each_callback(void* elem, dtl_size_t idx, void* user_data) {
    py::gil_scoped_acquire gil;

    auto* ctx = static_cast<PythonCallbackContext*>(user_data);

    // Create a numpy scalar from the element
    py::module_ np = py::module_::import("numpy");
    py::object value;

    switch (ctx->dtype) {
        case DTL_DTYPE_FLOAT64:
            value = py::cast(*static_cast<double*>(elem));
            break;
        case DTL_DTYPE_FLOAT32:
            value = py::cast(*static_cast<float*>(elem));
            break;
        case DTL_DTYPE_INT64:
            value = py::cast(*static_cast<int64_t*>(elem));
            break;
        case DTL_DTYPE_INT32:
            value = py::cast(*static_cast<int32_t*>(elem));
            break;
        case DTL_DTYPE_INT16:
            value = py::cast(*static_cast<int16_t*>(elem));
            break;
        case DTL_DTYPE_INT8:
            value = py::cast(*static_cast<int8_t*>(elem));
            break;
        case DTL_DTYPE_UINT64:
            value = py::cast(*static_cast<uint64_t*>(elem));
            break;
        case DTL_DTYPE_UINT32:
            value = py::cast(*static_cast<uint32_t*>(elem));
            break;
        case DTL_DTYPE_UINT16:
            value = py::cast(*static_cast<uint16_t*>(elem));
            break;
        case DTL_DTYPE_UINT8:
            value = py::cast(*static_cast<uint8_t*>(elem));
            break;
        default:
            throw std::runtime_error("Unsupported dtype in for_each callback");
    }

    if (ctx->with_index) {
        ctx->func(value, idx);
    } else {
        ctx->func(value);
    }
}

// Predicate callback that invokes Python function
int python_predicate_callback(const void* elem, void* user_data) {
    py::gil_scoped_acquire gil;

    auto* ctx = static_cast<PythonCallbackContext*>(user_data);

    py::object value;
    switch (ctx->dtype) {
        case DTL_DTYPE_FLOAT64:
            value = py::cast(*static_cast<const double*>(elem));
            break;
        case DTL_DTYPE_FLOAT32:
            value = py::cast(*static_cast<const float*>(elem));
            break;
        case DTL_DTYPE_INT64:
            value = py::cast(*static_cast<const int64_t*>(elem));
            break;
        case DTL_DTYPE_INT32:
            value = py::cast(*static_cast<const int32_t*>(elem));
            break;
        case DTL_DTYPE_INT16:
            value = py::cast(*static_cast<const int16_t*>(elem));
            break;
        case DTL_DTYPE_INT8:
            value = py::cast(*static_cast<const int8_t*>(elem));
            break;
        case DTL_DTYPE_UINT64:
            value = py::cast(*static_cast<const uint64_t*>(elem));
            break;
        case DTL_DTYPE_UINT32:
            value = py::cast(*static_cast<const uint32_t*>(elem));
            break;
        case DTL_DTYPE_UINT16:
            value = py::cast(*static_cast<const uint16_t*>(elem));
            break;
        case DTL_DTYPE_UINT8:
            value = py::cast(*static_cast<const uint8_t*>(elem));
            break;
        default:
            throw std::runtime_error("Unsupported dtype in predicate callback");
    }

    py::object result = ctx->func(value);
    return result.cast<bool>() ? 1 : 0;
}

// Comparator callback that invokes Python function
int python_comparator_callback(const void* a, const void* b, void* user_data) {
    py::gil_scoped_acquire gil;

    auto* ctx = static_cast<PythonCallbackContext*>(user_data);

    py::object val_a, val_b;
    switch (ctx->dtype) {
        case DTL_DTYPE_FLOAT64:
            val_a = py::cast(*static_cast<const double*>(a));
            val_b = py::cast(*static_cast<const double*>(b));
            break;
        case DTL_DTYPE_FLOAT32:
            val_a = py::cast(*static_cast<const float*>(a));
            val_b = py::cast(*static_cast<const float*>(b));
            break;
        case DTL_DTYPE_INT64:
            val_a = py::cast(*static_cast<const int64_t*>(a));
            val_b = py::cast(*static_cast<const int64_t*>(b));
            break;
        case DTL_DTYPE_INT32:
            val_a = py::cast(*static_cast<const int32_t*>(a));
            val_b = py::cast(*static_cast<const int32_t*>(b));
            break;
        default:
            val_a = py::cast(*static_cast<const double*>(a));
            val_b = py::cast(*static_cast<const double*>(b));
            break;
    }

    py::object result = ctx->func(val_a, val_b);
    return result.cast<int>();
}

// Transform callback that invokes Python function and writes result
void python_transform_callback(const void* input, void* output, dtl_size_t idx, void* user_data) {
    py::gil_scoped_acquire gil;

    auto* ctx = static_cast<PythonCallbackContext*>(user_data);

    // Create a Python value from the input element
    py::object value;
    switch (ctx->dtype) {
        case DTL_DTYPE_FLOAT64:
            value = py::cast(*static_cast<const double*>(input));
            break;
        case DTL_DTYPE_FLOAT32:
            value = py::cast(*static_cast<const float*>(input));
            break;
        case DTL_DTYPE_INT64:
            value = py::cast(*static_cast<const int64_t*>(input));
            break;
        case DTL_DTYPE_INT32:
            value = py::cast(*static_cast<const int32_t*>(input));
            break;
        case DTL_DTYPE_INT16:
            value = py::cast(*static_cast<const int16_t*>(input));
            break;
        case DTL_DTYPE_INT8:
            value = py::cast(*static_cast<const int8_t*>(input));
            break;
        case DTL_DTYPE_UINT64:
            value = py::cast(*static_cast<const uint64_t*>(input));
            break;
        case DTL_DTYPE_UINT32:
            value = py::cast(*static_cast<const uint32_t*>(input));
            break;
        case DTL_DTYPE_UINT16:
            value = py::cast(*static_cast<const uint16_t*>(input));
            break;
        case DTL_DTYPE_UINT8:
            value = py::cast(*static_cast<const uint8_t*>(input));
            break;
        default:
            throw std::runtime_error("Unsupported dtype in transform callback");
    }

    // Call the Python function and get the result
    py::object result = ctx->func(value);

    // Write the result back to the output element
    switch (ctx->dtype) {
        case DTL_DTYPE_FLOAT64:
            *static_cast<double*>(output) = result.cast<double>();
            break;
        case DTL_DTYPE_FLOAT32:
            *static_cast<float*>(output) = result.cast<float>();
            break;
        case DTL_DTYPE_INT64:
            *static_cast<int64_t*>(output) = result.cast<int64_t>();
            break;
        case DTL_DTYPE_INT32:
            *static_cast<int32_t*>(output) = result.cast<int32_t>();
            break;
        case DTL_DTYPE_INT16:
            *static_cast<int16_t*>(output) = result.cast<int16_t>();
            break;
        case DTL_DTYPE_INT8:
            *static_cast<int8_t*>(output) = result.cast<int8_t>();
            break;
        case DTL_DTYPE_UINT64:
            *static_cast<uint64_t*>(output) = result.cast<uint64_t>();
            break;
        case DTL_DTYPE_UINT32:
            *static_cast<uint32_t*>(output) = result.cast<uint32_t>();
            break;
        case DTL_DTYPE_UINT16:
            *static_cast<uint16_t*>(output) = result.cast<uint16_t>();
            break;
        case DTL_DTYPE_UINT8:
            *static_cast<uint8_t*>(output) = result.cast<uint8_t>();
            break;
        default:
            throw std::runtime_error("Unsupported dtype in transform callback output");
    }
}

}  // namespace

// ============================================================================
// Module Binding
// ============================================================================

void bind_algorithms(py::module_& m) {

    // ========================================================================
    // For-Each Operations
    // ========================================================================

    m.def("for_each_vector",
        [](py::object vec_obj, py::function func, bool with_index) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{func, dtype, with_index};

            py::gil_scoped_release release;
            dtl_status status = dtl_for_each_vector(vec, python_for_each_callback, &ctx);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("func"),
        py::arg("with_index") = false,
        R"doc(
Apply a function to each local element of a vector.

Args:
    vec: DistributedVector to iterate over
    func: Function to apply. If with_index=False, called as func(value).
          If with_index=True, called as func(value, index).
    with_index: Whether to pass index to function (default: False)

Example:
    >>> dtl.for_each_vector(vec, lambda x: print(x))
    >>> dtl.for_each_vector(vec, lambda x, i: print(f"[{i}] = {x}"), with_index=True)
)doc");

    m.def("for_each_array",
        [](py::object arr_obj, py::function func, bool with_index) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{func, dtype, with_index};

            py::gil_scoped_release release;
            dtl_status status = dtl_for_each_array(arr, python_for_each_callback, &ctx);
            check_status(status);
        },
        py::arg("arr"),
        py::arg("func"),
        py::arg("with_index") = false,
        R"doc(
Apply a function to each local element of an array.

Args:
    arr: DistributedArray to iterate over
    func: Function to apply
    with_index: Whether to pass index to function (default: False)
)doc");

    // ========================================================================
    // Copy/Fill Operations
    // ========================================================================

    m.def("copy_vector",
        [](py::object src_obj, py::object dst_obj) {
            dtl_vector_t src = get_native_vector(src_obj);
            dtl_vector_t dst = get_native_vector(dst_obj);
            dtl_status status = dtl_copy_vector(src, dst);
            check_status(status);
        },
        py::arg("src"),
        py::arg("dst"),
        R"doc(
Copy local data from source vector to destination vector.

Both vectors must have the same dtype and local size.

Args:
    src: Source DistributedVector
    dst: Destination DistributedVector
)doc");

    m.def("copy_array",
        [](py::object src_obj, py::object dst_obj) {
            dtl_array_t src = get_native_array(src_obj);
            dtl_array_t dst = get_native_array(dst_obj);
            dtl_status status = dtl_copy_array(src, dst);
            check_status(status);
        },
        py::arg("src"),
        py::arg("dst"),
        R"doc(
Copy local data from source array to destination array.

Args:
    src: Source DistributedArray
    dst: Destination DistributedArray
)doc");

    m.def("fill_vector",
        [](py::object vec_obj, py::object value_obj) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            // Convert Python value to appropriate C type
            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(value_ptr) = value_obj.cast<uint64_t>();
                    break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(value_ptr) = value_obj.cast<uint32_t>();
                    break;
                case DTL_DTYPE_UINT8:
                    *static_cast<uint8_t*>(value_ptr) = value_obj.cast<uint8_t>();
                    break;
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(value_ptr) = value_obj.cast<int8_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            dtl_status status = dtl_fill_vector(vec, value_ptr);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("value"),
        R"doc(
Fill all local elements of a vector with a value.

Args:
    vec: DistributedVector to fill
    value: Value to fill with

Example:
    >>> dtl.fill_vector(vec, 42.0)
)doc");

    m.def("fill_array",
        [](py::object arr_obj, py::object value_obj) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(value_ptr) = value_obj.cast<uint64_t>();
                    break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(value_ptr) = value_obj.cast<uint32_t>();
                    break;
                case DTL_DTYPE_UINT8:
                    *static_cast<uint8_t*>(value_ptr) = value_obj.cast<uint8_t>();
                    break;
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(value_ptr) = value_obj.cast<int8_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            dtl_status status = dtl_fill_array(arr, value_ptr);
            check_status(status);
        },
        py::arg("arr"),
        py::arg("value"),
        R"doc(
Fill all local elements of an array with a value.

Args:
    arr: DistributedArray to fill
    value: Value to fill with
)doc");

    // ========================================================================
    // Find Operations
    // ========================================================================

    m.def("find_vector",
        [](py::object vec_obj, py::object value_obj) -> py::object {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(value_ptr) = value_obj.cast<uint64_t>();
                    break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(value_ptr) = value_obj.cast<uint32_t>();
                    break;
                case DTL_DTYPE_UINT8:
                    *static_cast<uint8_t*>(value_ptr) = value_obj.cast<uint8_t>();
                    break;
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(value_ptr) = value_obj.cast<int8_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            dtl_index_t idx = dtl_find_vector(vec, value_ptr);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("vec"),
        py::arg("value"),
        R"doc(
Find the first occurrence of a value in the local partition.

Args:
    vec: DistributedVector to search
    value: Value to find

Returns:
    Local index of first match, or None if not found
)doc");

    m.def("find_if_vector",
        [](py::object vec_obj, py::function pred) -> py::object {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{pred, dtype, false};

            dtl_index_t idx = dtl_find_if_vector(vec, python_predicate_callback, &ctx);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("vec"),
        py::arg("predicate"),
        R"doc(
Find the first element satisfying a predicate in the local partition.

Args:
    vec: DistributedVector to search
    predicate: Function that returns True for matching elements

Returns:
    Local index of first match, or None if not found

Example:
    >>> idx = dtl.find_if_vector(vec, lambda x: x > 10)
)doc");

    m.def("find_array",
        [](py::object arr_obj, py::object value_obj) -> py::object {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                case DTL_DTYPE_UINT64:
                    *static_cast<uint64_t*>(value_ptr) = value_obj.cast<uint64_t>();
                    break;
                case DTL_DTYPE_UINT32:
                    *static_cast<uint32_t*>(value_ptr) = value_obj.cast<uint32_t>();
                    break;
                case DTL_DTYPE_UINT8:
                    *static_cast<uint8_t*>(value_ptr) = value_obj.cast<uint8_t>();
                    break;
                case DTL_DTYPE_INT8:
                    *static_cast<int8_t*>(value_ptr) = value_obj.cast<int8_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            dtl_index_t idx = dtl_find_array(arr, value_ptr);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("arr"),
        py::arg("value"),
        R"doc(
Find the first occurrence of a value in the local partition.

Args:
    arr: DistributedArray to search
    value: Value to find

Returns:
    Local index of first match, or None if not found
)doc");

    m.def("find_if_array",
        [](py::object arr_obj, py::function pred) -> py::object {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{pred, dtype, false};

            dtl_index_t idx = dtl_find_if_array(arr, python_predicate_callback, &ctx);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("arr"),
        py::arg("predicate"),
        R"doc(
Find the first element satisfying a predicate in the local partition.

Args:
    arr: DistributedArray to search
    predicate: Function that returns True for matching elements

Returns:
    Local index of first match, or None if not found
)doc");

    // ========================================================================
    // Count Operations
    // ========================================================================

    m.def("count_vector",
        [](py::object vec_obj, py::object value_obj) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            return dtl_count_vector(vec, value_ptr);
        },
        py::arg("vec"),
        py::arg("value"),
        R"doc(
Count occurrences of a value in the local partition.

Args:
    vec: DistributedVector to search
    value: Value to count

Returns:
    Number of matching elements
)doc");

    m.def("count_if_vector",
        [](py::object vec_obj, py::function pred) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{pred, dtype, false};

            return dtl_count_if_vector(vec, python_predicate_callback, &ctx);
        },
        py::arg("vec"),
        py::arg("predicate"),
        R"doc(
Count elements satisfying a predicate in the local partition.

Args:
    vec: DistributedVector to search
    predicate: Function that returns True for matching elements

Returns:
    Number of matching elements

Example:
    >>> n = dtl.count_if_vector(vec, lambda x: x > 0)
)doc");

    m.def("count_array",
        [](py::object arr_obj, py::object value_obj) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            std::aligned_storage_t<8, 8> value_storage;
            void* value_ptr = &value_storage;

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
                case DTL_DTYPE_FLOAT32:
                    *static_cast<float*>(value_ptr) = value_obj.cast<float>();
                    break;
                case DTL_DTYPE_INT64:
                    *static_cast<int64_t*>(value_ptr) = value_obj.cast<int64_t>();
                    break;
                case DTL_DTYPE_INT32:
                    *static_cast<int32_t*>(value_ptr) = value_obj.cast<int32_t>();
                    break;
                default:
                    *static_cast<double*>(value_ptr) = value_obj.cast<double>();
                    break;
            }

            return dtl_count_array(arr, value_ptr);
        },
        py::arg("arr"),
        py::arg("value"),
        R"doc(
Count occurrences of a value in the local partition.

Args:
    arr: DistributedArray to search
    value: Value to count

Returns:
    Number of matching elements
)doc");

    m.def("count_if_array",
        [](py::object arr_obj, py::function pred) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{pred, dtype, false};

            return dtl_count_if_array(arr, python_predicate_callback, &ctx);
        },
        py::arg("arr"),
        py::arg("predicate"),
        R"doc(
Count elements satisfying a predicate in the local partition.

Args:
    arr: DistributedArray to search
    predicate: Function that returns True for matching elements

Returns:
    Number of matching elements
)doc");

    // ========================================================================
    // Local Reduction Operations
    // ========================================================================

    m.def("reduce_local_vector",
        [](py::object vec_obj, const std::string& op) -> py::object {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            std::aligned_storage_t<8, 8> result_storage;
            void* result_ptr = &result_storage;

            dtl_status status = dtl_reduce_local_vector(vec, reduce_op, result_ptr);
            check_status(status);

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    return py::cast(*static_cast<double*>(result_ptr));
                case DTL_DTYPE_FLOAT32:
                    return py::cast(*static_cast<float*>(result_ptr));
                case DTL_DTYPE_INT64:
                    return py::cast(*static_cast<int64_t*>(result_ptr));
                case DTL_DTYPE_INT32:
                    return py::cast(*static_cast<int32_t*>(result_ptr));
                default:
                    return py::cast(*static_cast<double*>(result_ptr));
            }
        },
        py::arg("vec"),
        py::arg("op") = "sum",
        R"doc(
Reduce local elements of a vector using a built-in operation.

For distributed reduction (across all ranks), use allreduce() or reduce()
from dtl collective operations.

Args:
    vec: DistributedVector to reduce
    op: Reduction operation ("sum", "prod", "min", "max")

Returns:
    Reduced value for local partition

Example:
    >>> local_sum = dtl.reduce_local_vector(vec, op="sum")
    >>> global_sum = dtl.allreduce(ctx, np.array([local_sum]), op="sum")
)doc");

    m.def("reduce_local_array",
        [](py::object arr_obj, const std::string& op) -> py::object {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            std::aligned_storage_t<8, 8> result_storage;
            void* result_ptr = &result_storage;

            dtl_status status = dtl_reduce_local_array(arr, reduce_op, result_ptr);
            check_status(status);

            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    return py::cast(*static_cast<double*>(result_ptr));
                case DTL_DTYPE_FLOAT32:
                    return py::cast(*static_cast<float*>(result_ptr));
                case DTL_DTYPE_INT64:
                    return py::cast(*static_cast<int64_t*>(result_ptr));
                case DTL_DTYPE_INT32:
                    return py::cast(*static_cast<int32_t*>(result_ptr));
                default:
                    return py::cast(*static_cast<double*>(result_ptr));
            }
        },
        py::arg("arr"),
        py::arg("op") = "sum",
        R"doc(
Reduce local elements of an array using a built-in operation.

Args:
    arr: DistributedArray to reduce
    op: Reduction operation ("sum", "prod", "min", "max")

Returns:
    Reduced value for local partition
)doc");

    // ========================================================================
    // Sorting Operations
    // ========================================================================

    m.def("sort_vector",
        [](py::object vec_obj, bool reverse) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_status status;
            {
                py::gil_scoped_release release;
                if (reverse) {
                    status = dtl_sort_vector_descending(vec);
                } else {
                    status = dtl_sort_vector(vec);
                }
            }
            check_status(status);
        },
        py::arg("vec"),
        py::arg("reverse") = false,
        R"doc(
Sort local elements of a vector.

This is a local operation - only the local partition is sorted.

Args:
    vec: DistributedVector to sort
    reverse: If True, sort descending (default: False)
)doc");

    m.def("sort_vector_func",
        [](py::object vec_obj, py::function key_func) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            // Create a comparator that uses the key function
            auto cmp = [&key_func](py::object a, py::object b) -> int {
                py::object key_a = key_func(a);
                py::object key_b = key_func(b);
                if (key_a < key_b) return -1;
                if (key_a > key_b) return 1;
                return 0;
            };

            PythonCallbackContext ctx{py::cpp_function(cmp), dtype, false};

            py::gil_scoped_release release;
            dtl_status status = dtl_sort_vector_func(vec, python_comparator_callback, &ctx);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("key"),
        R"doc(
Sort local elements of a vector using a key function.

Args:
    vec: DistributedVector to sort
    key: Function that extracts comparison key from each element

Example:
    >>> dtl.sort_vector_func(vec, key=lambda x: abs(x))
)doc");

    m.def("sort_array",
        [](py::object arr_obj, bool reverse) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_status status;
            {
                py::gil_scoped_release release;
                if (reverse) {
                    status = dtl_sort_array_descending(arr);
                } else {
                    status = dtl_sort_array(arr);
                }
            }
            check_status(status);
        },
        py::arg("arr"),
        py::arg("reverse") = false,
        R"doc(
Sort local elements of an array.

This is a local operation - only the local partition is sorted.

Args:
    arr: DistributedArray to sort
    reverse: If True, sort descending (default: False)
)doc");

    // ========================================================================
    // Min/Max Operations
    // ========================================================================

    m.def("minmax_vector",
        [](py::object vec_obj) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            std::aligned_storage_t<8, 8> min_storage, max_storage;
            void* min_ptr = &min_storage;
            void* max_ptr = &max_storage;

            dtl_status status = dtl_minmax_vector(vec, min_ptr, max_ptr);
            check_status(status);

            py::object min_val, max_val;
            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    min_val = py::cast(*static_cast<double*>(min_ptr));
                    max_val = py::cast(*static_cast<double*>(max_ptr));
                    break;
                case DTL_DTYPE_FLOAT32:
                    min_val = py::cast(*static_cast<float*>(min_ptr));
                    max_val = py::cast(*static_cast<float*>(max_ptr));
                    break;
                case DTL_DTYPE_INT64:
                    min_val = py::cast(*static_cast<int64_t*>(min_ptr));
                    max_val = py::cast(*static_cast<int64_t*>(max_ptr));
                    break;
                case DTL_DTYPE_INT32:
                    min_val = py::cast(*static_cast<int32_t*>(min_ptr));
                    max_val = py::cast(*static_cast<int32_t*>(max_ptr));
                    break;
                default:
                    min_val = py::cast(*static_cast<double*>(min_ptr));
                    max_val = py::cast(*static_cast<double*>(max_ptr));
                    break;
            }

            return py::make_tuple(min_val, max_val);
        },
        py::arg("vec"),
        R"doc(
Find minimum and maximum values in local vector.

Args:
    vec: DistributedVector to search

Returns:
    Tuple of (min_value, max_value)

Example:
    >>> min_val, max_val = dtl.minmax_vector(vec)
)doc");

    m.def("minmax_array",
        [](py::object arr_obj) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            std::aligned_storage_t<8, 8> min_storage, max_storage;
            void* min_ptr = &min_storage;
            void* max_ptr = &max_storage;

            dtl_status status = dtl_minmax_array(arr, min_ptr, max_ptr);
            check_status(status);

            py::object min_val, max_val;
            switch (dtype) {
                case DTL_DTYPE_FLOAT64:
                    min_val = py::cast(*static_cast<double*>(min_ptr));
                    max_val = py::cast(*static_cast<double*>(max_ptr));
                    break;
                case DTL_DTYPE_FLOAT32:
                    min_val = py::cast(*static_cast<float*>(min_ptr));
                    max_val = py::cast(*static_cast<float*>(max_ptr));
                    break;
                case DTL_DTYPE_INT64:
                    min_val = py::cast(*static_cast<int64_t*>(min_ptr));
                    max_val = py::cast(*static_cast<int64_t*>(max_ptr));
                    break;
                case DTL_DTYPE_INT32:
                    min_val = py::cast(*static_cast<int32_t*>(min_ptr));
                    max_val = py::cast(*static_cast<int32_t*>(max_ptr));
                    break;
                default:
                    min_val = py::cast(*static_cast<double*>(min_ptr));
                    max_val = py::cast(*static_cast<double*>(max_ptr));
                    break;
            }

            return py::make_tuple(min_val, max_val);
        },
        py::arg("arr"),
        R"doc(
Find minimum and maximum values in local array.

Args:
    arr: DistributedArray to search

Returns:
    Tuple of (min_value, max_value)
)doc");

    // ========================================================================
    // Transform Operations
    // ========================================================================

    m.def("transform_vector",
        [](py::object vec_obj, py::function func) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{func, dtype, false};

            py::gil_scoped_release release;
            dtl_status status = dtl_transform_vector(vec, vec, python_transform_callback, &ctx);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("func"),
        R"doc(
Apply a transformation function to each element of a distributed vector.

The function is called with each element's value and should return
the transformed value. The vector is modified in-place.

Args:
    vec: DistributedVector to transform
    func: Function that takes a value and returns the transformed value

Example:
    >>> dtl.transform_vector(vec, lambda x: x * 2)
    >>> dtl.transform_vector(vec, lambda x: x ** 2 + 1)
)doc");

    m.def("transform_array",
        [](py::object arr_obj, py::function func) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{func, dtype, false};

            py::gil_scoped_release release;
            dtl_status status = dtl_transform_array(arr, arr, python_transform_callback, &ctx);
            check_status(status);
        },
        py::arg("arr"),
        py::arg("func"),
        R"doc(
Apply a transformation function to each element of a distributed array.

The function is called with each element's value and should return
the transformed value. The array is modified in-place.

Args:
    arr: DistributedArray to transform
    func: Function that takes a value and returns the transformed value

Example:
    >>> dtl.transform_array(arr, lambda x: x * 2)
)doc");

    // ========================================================================
    // Scan / Prefix Operations
    // ========================================================================

    m.def("inclusive_scan_vector",
        [](py::object vec_obj, const std::string& op) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            py::gil_scoped_release release;
            dtl_status status = dtl_inclusive_scan_vector(vec, reduce_op);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("op") = "sum",
        R"doc(
Compute inclusive prefix scan of a distributed vector.

Each element i is replaced by the reduction of elements 0..i.
The vector is modified in-place.

Args:
    vec: DistributedVector to scan
    op: Reduction operation ("sum", "prod", "min", "max")

Example:
    >>> # [1, 2, 3, 4] -> [1, 3, 6, 10] with op="sum"
    >>> dtl.inclusive_scan_vector(vec, op="sum")
)doc");

    m.def("exclusive_scan_vector",
        [](py::object vec_obj, const std::string& op) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            py::gil_scoped_release release;
            dtl_status status = dtl_exclusive_scan_vector(vec, reduce_op);
            check_status(status);
        },
        py::arg("vec"),
        py::arg("op") = "sum",
        R"doc(
Compute exclusive prefix scan of a distributed vector.

Each element i is replaced by the reduction of elements 0..i-1.
The first element is set to the identity for the operation
(0 for sum, 1 for product). The vector is modified in-place.

Args:
    vec: DistributedVector to scan
    op: Reduction operation ("sum", "prod", "min", "max")

Example:
    >>> # [1, 2, 3, 4] -> [0, 1, 3, 6] with op="sum"
    >>> dtl.exclusive_scan_vector(vec, op="sum")
)doc");

    m.def("inclusive_scan_array",
        [](py::object arr_obj, const std::string& op) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            py::gil_scoped_release release;
            dtl_status status = dtl_inclusive_scan_array(arr, reduce_op);
            check_status(status);
        },
        py::arg("arr"),
        py::arg("op") = "sum",
        R"doc(
Compute inclusive prefix scan of a distributed array.

Each element i is replaced by the reduction of elements 0..i.
The array is modified in-place.

Args:
    arr: DistributedArray to scan
    op: Reduction operation ("sum", "prod", "min", "max")
)doc");

    m.def("exclusive_scan_array",
        [](py::object arr_obj, const std::string& op) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_reduce_op reduce_op = string_to_reduce_op(op);

            py::gil_scoped_release release;
            dtl_status status = dtl_exclusive_scan_array(arr, reduce_op);
            check_status(status);
        },
        py::arg("arr"),
        py::arg("op") = "sum",
        R"doc(
Compute exclusive prefix scan of a distributed array.

Each element i is replaced by the reduction of elements 0..i-1.
The first element is set to the identity for the operation.
The array is modified in-place.

Args:
    arr: DistributedArray to scan
    op: Reduction operation ("sum", "prod", "min", "max")
)doc");

    // ========================================================================
    // Predicate Query Operations (Phase 16)
    // ========================================================================

    m.def("all_of_vector",
        [](py::object vec_obj, py::function pred) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_all_of_vector(vec, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("all_of_vector failed");
            }
            return result != 0;
        },
        py::arg("vec"),
        py::arg("predicate"),
        R"doc(
Check if all local elements satisfy a predicate.

Args:
    vec: DistributedVector to check
    predicate: Function that returns True/False for each element

Returns:
    True if all local elements satisfy the predicate (or vector is empty)

Example:
    >>> all_positive = dtl.all_of_vector(vec, lambda x: x > 0)
)doc");

    m.def("any_of_vector",
        [](py::object vec_obj, py::function pred) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_any_of_vector(vec, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("any_of_vector failed");
            }
            return result != 0;
        },
        py::arg("vec"),
        py::arg("predicate"),
        R"doc(
Check if any local element satisfies a predicate.

Args:
    vec: DistributedVector to check
    predicate: Function that returns True/False for each element

Returns:
    True if any local element satisfies the predicate

Example:
    >>> has_negative = dtl.any_of_vector(vec, lambda x: x < 0)
)doc");

    m.def("none_of_vector",
        [](py::object vec_obj, py::function pred) {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_dtype dtype = dtl_vector_dtype(vec);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_none_of_vector(vec, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("none_of_vector failed");
            }
            return result != 0;
        },
        py::arg("vec"),
        py::arg("predicate"),
        R"doc(
Check if no local elements satisfy a predicate.

Args:
    vec: DistributedVector to check
    predicate: Function that returns True/False for each element

Returns:
    True if no local elements satisfy the predicate (or vector is empty)

Example:
    >>> no_nans = dtl.none_of_vector(vec, lambda x: x != x)
)doc");

    m.def("all_of_array",
        [](py::object arr_obj, py::function pred) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_all_of_array(arr, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("all_of_array failed");
            }
            return result != 0;
        },
        py::arg("arr"),
        py::arg("predicate"),
        R"doc(
Check if all local elements satisfy a predicate.

Args:
    arr: DistributedArray to check
    predicate: Function that returns True/False for each element

Returns:
    True if all local elements satisfy the predicate (or array is empty)
)doc");

    m.def("any_of_array",
        [](py::object arr_obj, py::function pred) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_any_of_array(arr, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("any_of_array failed");
            }
            return result != 0;
        },
        py::arg("arr"),
        py::arg("predicate"),
        R"doc(
Check if any local element satisfies a predicate.

Args:
    arr: DistributedArray to check
    predicate: Function that returns True/False for each element

Returns:
    True if any local element satisfies the predicate
)doc");

    m.def("none_of_array",
        [](py::object arr_obj, py::function pred) {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_dtype dtype = dtl_array_dtype(arr);

            PythonCallbackContext ctx{pred, dtype, false};

            int result = dtl_none_of_array(arr, python_predicate_callback, &ctx);
            if (result < 0) {
                throw std::runtime_error("none_of_array failed");
            }
            return result != 0;
        },
        py::arg("arr"),
        py::arg("predicate"),
        R"doc(
Check if no local elements satisfy a predicate.

Args:
    arr: DistributedArray to check
    predicate: Function that returns True/False for each element

Returns:
    True if no local elements satisfy the predicate (or array is empty)
)doc");

    // ========================================================================
    // Extrema Element Operations (Phase 16)
    // ========================================================================

    m.def("min_element_vector",
        [](py::object vec_obj) -> py::object {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_index_t idx = dtl_min_element_vector(vec);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("vec"),
        R"doc(
Find the index of the minimum element in the local partition.

Args:
    vec: DistributedVector to search

Returns:
    Local index of the minimum element, or None if empty

Example:
    >>> idx = dtl.min_element_vector(vec)
)doc");

    m.def("max_element_vector",
        [](py::object vec_obj) -> py::object {
            dtl_vector_t vec = get_native_vector(vec_obj);
            dtl_index_t idx = dtl_max_element_vector(vec);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("vec"),
        R"doc(
Find the index of the maximum element in the local partition.

Args:
    vec: DistributedVector to search

Returns:
    Local index of the maximum element, or None if empty

Example:
    >>> idx = dtl.max_element_vector(vec)
)doc");

    m.def("min_element_array",
        [](py::object arr_obj) -> py::object {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_index_t idx = dtl_min_element_array(arr);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("arr"),
        R"doc(
Find the index of the minimum element in the local partition.

Args:
    arr: DistributedArray to search

Returns:
    Local index of the minimum element, or None if empty
)doc");

    m.def("max_element_array",
        [](py::object arr_obj) -> py::object {
            dtl_array_t arr = get_native_array(arr_obj);
            dtl_index_t idx = dtl_max_element_array(arr);
            if (idx < 0) {
                return py::none();
            }
            return py::cast(idx);
        },
        py::arg("arr"),
        R"doc(
Find the index of the maximum element in the local partition.

Args:
    arr: DistributedArray to search

Returns:
    Local index of the maximum element, or None if empty
)doc");
}
