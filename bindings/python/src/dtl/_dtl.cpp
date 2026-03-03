// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file _dtl.cpp
 * @brief DTL Python bindings - Main module definition
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

#include "status_exception.hpp"

namespace py = pybind11;

// Forward declarations
void bind_core(py::module_& m);
void bind_containers(py::module_& m);
void bind_collective(py::module_& m);
void bind_policies(py::module_& m);
void bind_algorithms(py::module_& m);
void bind_rma(py::module_& m);
void init_mpmd(py::module_& m);
void init_topology(py::module_& m);
void init_futures(py::module_& m);
void init_remote(py::module_& m);

PYBIND11_MODULE(_dtl, m) {
    m.doc() = "DTL (Distributed Template Library) Python bindings";

    // Version info
    m.attr("__version__") = DTL_VERSION_PYTHON;
    m.attr("version_major") = DTL_VERSION_MAJOR;
    m.attr("version_minor") = DTL_VERSION_MINOR;
    m.attr("version_patch") = DTL_VERSION_PATCH;

    // Feature detection functions
    m.def("has_mpi", []() { return dtl_has_mpi() != 0; },
          "Check if MPI backend is available");
    m.def("has_cuda", []() { return dtl_has_cuda() != 0; },
          "Check if CUDA backend is available");
    m.def("has_hip", []() { return dtl_has_hip() != 0; },
          "Check if HIP backend is available");
    m.def("has_nccl", []() { return dtl_has_nccl() != 0; },
          "Check if NCCL is available");
    m.def("has_shmem", []() { return dtl_has_shmem() != 0; },
          "Check if OpenSHMEM backend is available");

    // Register custom exception types from dtl Python package
    dtl::python::register_dtl_exceptions(m);

    // Bind components
    bind_core(m);
    bind_policies(m);
    bind_containers(m);
    bind_collective(m);
    bind_algorithms(m);
    bind_rma(m);
    init_mpmd(m);
    init_topology(m);
    init_futures(m);
    init_remote(m);
}
