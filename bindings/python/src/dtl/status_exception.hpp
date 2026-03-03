// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <pybind11/pybind11.h>

#include <dtl/bindings/c/dtl_status.h>

#include <string>

namespace dtl::python {

namespace py = pybind11;

// =============================================================================
// Custom exception type registry
// =============================================================================
// These are set during module initialization (register_dtl_exceptions) and
// point to the Python-side exception classes defined in dtl/__init__.py.

struct dtl_exception_types {
    PyObject* dtl_error = nullptr;
    PyObject* communication_error = nullptr;
    PyObject* memory_error = nullptr;
    PyObject* bounds_error = nullptr;
    PyObject* invalid_argument_error = nullptr;
    PyObject* backend_error = nullptr;
};

inline dtl_exception_types& exception_registry() {
    static dtl_exception_types types;
    return types;
}

/// Register the custom DTL exception types from the Python dtl module.
/// Must be called during pybind11 module initialization.
inline void register_dtl_exceptions(py::module_& m) {
    // Import the dtl Python package to access the exception classes
    py::module_ dtl_mod = py::module_::import("dtl");

    auto& reg = exception_registry();
    reg.dtl_error = dtl_mod.attr("DTLError").ptr();
    reg.communication_error = dtl_mod.attr("CommunicationError").ptr();
    reg.memory_error = dtl_mod.attr("MemoryError").ptr();
    reg.bounds_error = dtl_mod.attr("BoundsError").ptr();
    reg.invalid_argument_error = dtl_mod.attr("InvalidArgumentError").ptr();
    reg.backend_error = dtl_mod.attr("BackendError").ptr();

    (void)m;  // module reference available if needed in the future
}

// =============================================================================
// Exception raising
// =============================================================================

[[noreturn]] inline void raise_python_exception(PyObject* exc_type,
                                                const std::string& message) {
    PyErr_SetString(exc_type, message.c_str());
    throw py::error_already_set();
}

inline void check_status_or_throw(dtl_status status) {
    if (!dtl_status_is_error(status)) {
        return;
    }

    const char* message = dtl_status_message(status);
    const char* code_name = dtl_status_name(status);
    const std::string detail = std::string(code_name) + ": " + message;

    auto& reg = exception_registry();

    // Map specific error codes to custom exception types
    switch (status) {
        case DTL_ERROR_INVALID_ARGUMENT:
        case DTL_ERROR_NULL_POINTER:
        case DTL_ERROR_PRECONDITION_FAILED:
        case DTL_ERROR_COLLECTIVE_PARTICIPATION:
        case DTL_ERROR_BACKEND_INVALID:
            if (reg.invalid_argument_error) {
                raise_python_exception(reg.invalid_argument_error, detail);
            }
            throw py::value_error(detail);

        case DTL_ERROR_OUT_OF_BOUNDS:
        case DTL_ERROR_INVALID_INDEX:
        case DTL_ERROR_INVALID_RANK:
        case DTL_ERROR_OUT_OF_RANGE:
            if (reg.bounds_error) {
                raise_python_exception(reg.bounds_error, detail);
            }
            throw py::index_error(detail);

        default:
            break;
    }

    // Map error categories to custom exception types
    const int category = dtl_status_category_code(status);
    switch (category) {
        case DTL_CATEGORY_COMMUNICATION:
            if (reg.communication_error) {
                raise_python_exception(reg.communication_error, detail);
            }
            raise_python_exception(PyExc_RuntimeError,
                                   std::string("CommunicationError: ") + detail);
        case DTL_CATEGORY_MEMORY:
            if (reg.memory_error) {
                raise_python_exception(reg.memory_error, detail);
            }
            raise_python_exception(PyExc_MemoryError, detail);
        case DTL_CATEGORY_BACKEND:
            if (reg.backend_error) {
                raise_python_exception(reg.backend_error, detail);
            }
            raise_python_exception(PyExc_RuntimeError,
                                   std::string("BackendError: ") + detail);
        case DTL_CATEGORY_ALGORITHM:
            if (reg.dtl_error) {
                raise_python_exception(reg.dtl_error,
                                       std::string("AlgorithmError: ") + detail);
            }
            raise_python_exception(PyExc_RuntimeError,
                                   std::string("AlgorithmError: ") + detail);
        default:
            if (reg.dtl_error) {
                raise_python_exception(reg.dtl_error, detail);
            }
            raise_python_exception(PyExc_RuntimeError,
                                   std::string("DTLError: ") + detail);
    }
}

}  // namespace dtl::python
