// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file policies.cpp
 * @brief DTL Python bindings - Policy enums and container options
 * @since 0.1.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dtl/bindings/c/dtl.h>

namespace py = pybind11;

void bind_policies(py::module_& m) {
    // ========================================================================
    // Partition Policy Enum
    // ========================================================================
    py::enum_<dtl_partition_policy>(m, "PartitionPolicy",
        "Partition policy types - how data is distributed across ranks")
        .value("BLOCK", DTL_PARTITION_BLOCK, "Block partition - contiguous chunks per rank (default)")
        .value("CYCLIC", DTL_PARTITION_CYCLIC, "Cyclic partition - round-robin distribution")
        .value("BLOCK_CYCLIC", DTL_PARTITION_BLOCK_CYCLIC, "Block-cyclic partition")
        .value("HASH", DTL_PARTITION_HASH, "Hash partition - hash-based distribution")
        .value("REPLICATED", DTL_PARTITION_REPLICATED, "Replicated - full copy on each rank")
        .export_values();

    // Also export as module-level constants for convenience
    m.attr("PARTITION_BLOCK") = DTL_PARTITION_BLOCK;
    m.attr("PARTITION_CYCLIC") = DTL_PARTITION_CYCLIC;
    m.attr("PARTITION_BLOCK_CYCLIC") = DTL_PARTITION_BLOCK_CYCLIC;
    m.attr("PARTITION_HASH") = DTL_PARTITION_HASH;
    m.attr("PARTITION_REPLICATED") = DTL_PARTITION_REPLICATED;

    // ========================================================================
    // Placement Policy Enum
    // ========================================================================
    py::enum_<dtl_placement_policy>(m, "PlacementPolicy",
        "Placement policy types - where data is stored (CPU vs GPU)")
        .value("HOST", DTL_PLACEMENT_HOST, "Host-only memory (CPU, default)")
        .value("DEVICE", DTL_PLACEMENT_DEVICE, "Device-only memory (GPU)")
        .value("UNIFIED", DTL_PLACEMENT_UNIFIED, "Unified/managed memory")
        .value("DEVICE_PREFERRED", DTL_PLACEMENT_DEVICE_PREFERRED, "Device-preferred with host fallback")
        .export_values();

    // Module-level constants
    m.attr("PLACEMENT_HOST") = DTL_PLACEMENT_HOST;
    m.attr("PLACEMENT_DEVICE") = DTL_PLACEMENT_DEVICE;
    m.attr("PLACEMENT_UNIFIED") = DTL_PLACEMENT_UNIFIED;
    m.attr("PLACEMENT_DEVICE_PREFERRED") = DTL_PLACEMENT_DEVICE_PREFERRED;

    // ========================================================================
    // Execution Policy Enum
    // ========================================================================
    py::enum_<dtl_execution_policy>(m, "ExecutionPolicy",
        "Execution policy types - how operations are executed")
        .value("SEQ", DTL_EXEC_SEQ, "Sequential execution (blocking, single-threaded)")
        .value("PAR", DTL_EXEC_PAR, "Parallel execution (blocking, multi-threaded)")
        .value("ASYNC", DTL_EXEC_ASYNC, "Asynchronous execution (non-blocking)")
        .export_values();

    // Module-level constants
    m.attr("EXEC_SEQ") = DTL_EXEC_SEQ;
    m.attr("EXEC_PAR") = DTL_EXEC_PAR;
    m.attr("EXEC_ASYNC") = DTL_EXEC_ASYNC;

    // ========================================================================
    // Utility Functions
    // ========================================================================
    m.def("partition_policy_name", &dtl_partition_policy_name,
          py::arg("policy"),
          "Get the name of a partition policy");

    m.def("placement_policy_name", &dtl_placement_policy_name,
          py::arg("policy"),
          "Get the name of a placement policy");

    m.def("execution_policy_name", &dtl_execution_policy_name,
          py::arg("policy"),
          "Get the name of an execution policy");

    m.def("placement_available", &dtl_placement_available,
          py::arg("policy"),
          R"doc(
Check if a placement policy is available in this build.

Some placements (device, unified, device_preferred) require CUDA support.

Args:
    policy: The placement policy to check

Returns:
    True if available, False otherwise
)doc");
}
