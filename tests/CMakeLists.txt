# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Test build configuration
# =============================================================================
# DTL test suite using GoogleTest
# =============================================================================

# -----------------------------------------------------------------------------
# Header Smoke Test (no dependencies - verifies library compiles)
# -----------------------------------------------------------------------------
# This test must come first and has no dependencies on GoogleTest.
# It verifies that #include <dtl/dtl.hpp> compiles cleanly.

add_executable(dtl_header_smoke smoke/test_header_smoke.cpp)
target_link_libraries(dtl_header_smoke PRIVATE DTL::dtl)
target_compile_features(dtl_header_smoke PRIVATE cxx_std_20)

# Apply strict warnings to catch issues
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dtl_header_smoke PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(dtl_header_smoke PRIVATE /W4)
endif()

# Add to CTest
add_test(NAME header_smoke COMMAND dtl_header_smoke)
set_tests_properties(header_smoke PROPERTIES
    LABELS "smoke"
    TIMEOUT 30
)

# Create smoke test directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/smoke)

# -----------------------------------------------------------------------------
# GoogleTest Setup
# -----------------------------------------------------------------------------

# Prefer system-installed GoogleTest when available to avoid network access in
# restricted environments (e.g., some CI/sandbox setups).
if(DTL_USE_SYSTEM_GTEST)
    find_package(GTest QUIET)
endif()

if(NOT TARGET GTest::gtest)
    include(FetchContent)

    # -------------------------------------------------------------------------
    # Fetch GoogleTest
    # -------------------------------------------------------------------------
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )

    # Prevent overriding the parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Don't install GTest when installing DTL
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)
endif()

# Include GoogleTest module for test discovery
include(GoogleTest)

# -----------------------------------------------------------------------------
# Test Utility Library (common test helpers)
# -----------------------------------------------------------------------------
add_library(dtl_test_utils INTERFACE)
target_link_libraries(dtl_test_utils INTERFACE
    DTL::dtl
    GTest::gtest
    GTest::gtest_main
)

# -----------------------------------------------------------------------------
# Unit Tests Target
# -----------------------------------------------------------------------------
add_executable(dtl_unit_tests)

# Set C++20 standard for tests
target_compile_features(dtl_unit_tests PRIVATE cxx_std_20)

# Link libraries
target_link_libraries(dtl_unit_tests PRIVATE
    dtl_test_utils
    DTL::dtl
    DTL::cpu
    GTest::gtest
    GTest::gtest_main
)

# C bindings helpers are only available when the C ABI is built
if(DTL_BUILD_C_BINDINGS)
    target_link_libraries(dtl_unit_tests PRIVATE dtl_c)
endif()

# Apply compiler warnings to test code
dtl_set_compiler_warnings(dtl_unit_tests)

# -----------------------------------------------------------------------------
# Test Source Files
# -----------------------------------------------------------------------------

# Core tests
target_sources(dtl_unit_tests PRIVATE
    unit/core/test_types.cpp
    unit/core/test_traits.cpp
    unit/core/test_concepts.cpp
    unit/core/test_thread_level.cpp
    unit/core/test_domain.cpp
    unit/core/test_context.cpp
    unit/core/test_runtime_dso.cpp
)

# Error tests
target_sources(dtl_unit_tests PRIVATE
    unit/error/test_status.cpp
    unit/error/test_result.cpp
    unit/error/test_error_types.cpp
    unit/error/test_error_type.cpp
    unit/error/test_collective_error.cpp
    unit/error/test_phase05_status_codes.cpp
    unit/error/test_phase05_failure_handler.cpp
    unit/error/test_phase05_result_api.cpp
)

# Serialization tests
target_sources(dtl_unit_tests PRIVATE
    unit/serialization/test_serializer.cpp
    unit/serialization/test_trivial_serialization.cpp
    unit/serialization/test_custom_serialization.cpp
    unit/serialization/test_cereal_adapter.cpp
    # unit/serialization/test_boost_serialization_adapter.cpp  # Disabled: Boost not available in this environment
)

# Backend concept tests
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_communicator_concept.cpp
    unit/backend/test_memory_space_concept.cpp
    unit/backend/test_executor_concept.cpp
    unit/backend/test_cpu_executor.cpp
    unit/backend/test_rma_concept.cpp
    unit/backend/test_mpi_rma.cpp
    unit/backend/test_mpi_request.cpp
    unit/backend/test_context_registry.cpp
    unit/backend/test_backend_concept_conformance.cpp
)

# Utility tests
target_sources(dtl_unit_tests PRIVATE
    unit/utility/test_resource.cpp
    unit/utility/test_phase05_debugging.cpp
)

# Policy tests (Phase 2)
target_sources(dtl_unit_tests PRIVATE
    unit/policies/test_policy_set.cpp
)

# Index translation tests (Phase 2)
target_sources(dtl_unit_tests PRIVATE
    unit/index/test_partition_map.cpp
    unit/index/test_index_translation.cpp
)

# Views tests (Phase 2)
target_sources(dtl_unit_tests PRIVATE
    unit/views/test_remote_ref.cpp
    unit/views/test_local_view_impl.cpp
    unit/views/test_batching_views.cpp
    unit/views/test_global_view.cpp
    unit/views/test_view_composition.cpp
)

# Container tests (Phase 2 + 10)
target_sources(dtl_unit_tests PRIVATE
    unit/containers/test_distributed_vector_impl.cpp
    unit/containers/test_distributed_array_impl.cpp
    unit/containers/test_distributed_tensor_impl.cpp
    unit/containers/test_distributed_vector.cpp
    unit/containers/test_distributed_map.cpp
    unit/containers/test_distributed_map_phase06.cpp
    unit/containers/test_distributed_span.cpp
)

# Algorithm tests (Phase 3)
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_algorithm_infrastructure.cpp
    unit/algorithms/test_for_each.cpp
    unit/algorithms/test_reduce.cpp
    unit/algorithms/test_transform.cpp
    unit/algorithms/test_sort.cpp
    unit/algorithms/test_copy_fill.cpp
    unit/algorithms/test_scan.cpp
)

# Algorithm correctness tests (Phase 06)
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_algorithm_traits_phase06.cpp
    unit/algorithms/test_phase02_api_semantic_split.cpp
    unit/algorithms/test_replace.cpp
    unit/algorithms/test_partial_sort.cpp
    unit/algorithms/test_nth_element.cpp
    unit/algorithms/test_unique.cpp
    unit/algorithms/test_predicates.cpp
    unit/algorithms/test_accumulate.cpp
    unit/algorithms/test_minmax.cpp
    unit/algorithms/test_find.cpp
    unit/algorithms/test_reduce_correctness.cpp
    unit/algorithms/test_transform_reduce_correctness.cpp
    unit/algorithms/test_binary_transform.cpp
)

# Algorithm expansion tests (R6)
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_adjacent_difference.cpp
    unit/algorithms/test_inner_product.cpp
    unit/algorithms/test_iota.cpp
    unit/algorithms/test_rotate.cpp
    unit/algorithms/test_partition_algorithm.cpp
    unit/algorithms/test_async_dispatch_types.cpp
)

# Futures tests (Phase 6)
target_sources(dtl_unit_tests PRIVATE
    unit/futures/test_progress.cpp
    unit/futures/test_progress_thread_safety.cpp
    unit/futures/test_when_all.cpp
    unit/futures/test_when_any.cpp
    unit/futures/test_continuation.cpp
    unit/futures/test_distributed_future.cpp
    unit/futures/test_cuda_event_future.cpp
    unit/futures/test_async_algorithms.cpp
)

# Futures tests (Phase 7 - Progress Engine Hardening)
target_sources(dtl_unit_tests PRIVATE
    unit/futures/test_progress_poll_api.cpp
    unit/futures/test_background_progress_mode.cpp
    unit/futures/test_callback_executor.cpp
    unit/futures/test_timeout_diagnostics.cpp
    integration/futures/test_progress_regressions.cpp
)

# Remote tests (Phase 6)
target_sources(dtl_unit_tests PRIVATE
    unit/remote/test_action.cpp
    unit/remote/test_argument_pack.cpp
)

# Remote tests (Phase 9 - Dynamic Handlers)
target_sources(dtl_unit_tests PRIVATE
    unit/remote/test_dynamic_handler_local.cpp
)

# Topology tests (Phase 6)
target_sources(dtl_unit_tests PRIVATE
    unit/topology/test_hardware.cpp
    unit/topology/test_proximity.cpp
    unit/topology/test_affinity.cpp
)

# RMA tests (Phase 7)
target_sources(dtl_unit_tests PRIVATE
    unit/communication/test_memory_window.cpp
    unit/communication/test_memory_window_data_transfer.cpp
    unit/communication/test_rma_operations.cpp
    unit/communication/test_rma_atomic.cpp
    unit/rma/test_window_guard.cpp
    unit/rma/test_async_rma.cpp
    unit/rma/test_remote_integration.cpp
)

# Probe operations tests (Phase 5 / V1.2.2 — Task 5.2)
target_sources(dtl_unit_tests PRIVATE
    unit/communication/test_probe_operations.cpp
)

# Communicator management tests (Phase 5)
target_sources(dtl_unit_tests PRIVATE
    unit/communication/test_communicator_split.cpp
)

# MPMD tests (Phase 11.5)
target_sources(dtl_unit_tests PRIVATE
    unit/mpmd/test_rank_group.cpp
    unit/mpmd/test_role_manager.cpp
)

# Phase 12A: Environment & Lifecycle tests
target_sources(dtl_unit_tests PRIVATE
    unit/core/test_environment_options.cpp
    unit/core/test_mpi_adopt_mode.cpp
    unit/core/test_environment_singleton.cpp
)

# Phase 12B: Core Algorithm & View Completion tests
target_sources(dtl_unit_tests PRIVATE
    unit/views/test_remote_ref_rma.cpp
    unit/algorithms/test_distributed_sort.cpp
    unit/communication/test_send_modes.cpp
    unit/communication/test_send_mode_wiring.cpp
    unit/communication/test_other_comm_ops.cpp
)

# Phase 12C: MPMD Completion tests (unit-level, no MPI required)
target_sources(dtl_unit_tests PRIVATE
    integration/test_mpmd_role_assignment.cpp
    integration/test_inter_group_comm.cpp
    integration/test_mpmd_workflow.cpp
)

# Phase 12D: GPU Backend Enhancement tests
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_hip_memory_transfer.cpp
    unit/backend/test_hip_memory_space.cpp
    unit/backend/test_nccl_communicator.cpp
    unit/memory/test_pinned_memory.cpp
    unit/memory/test_unified_memory_prefetch.cpp
    unit/memory/test_pointer_utils.cpp
)

# Phase 12E: Bindings Extension tests
if(DTL_BUILD_C_BINDINGS)
    target_sources(dtl_unit_tests PRIVATE
        unit/bindings/test_backend_bindings.cpp
    )
endif()

# Default communicator tests
target_sources(dtl_unit_tests PRIVATE
    unit/communication/test_default_communicator.cpp
)

# Phase 12F: Non-Functional Backend Remediation tests
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_shared_barrier.cpp
)

# Phase 03: Communication Correctness tests
target_sources(dtl_unit_tests PRIVATE
    unit/communication/test_collective_free_functions.cpp
    unit/communication/test_variable_size_collective_ops.cpp
    unit/communication/test_allreduce_inplace.cpp
)

# Phase 08: Container, View, and Iterator Resolution tests
target_sources(dtl_unit_tests PRIVATE
    # T01: Sync state for array and tensor
    unit/containers/test_array_sync_state.cpp
    unit/containers/test_tensor_sync_state.cpp
    # T02: Device affinity for array and tensor
    unit/containers/test_array_device_affinity.cpp
    unit/containers/test_tensor_device_affinity.cpp
    # T03: segmented_view tests
    unit/views/test_segmented_view.cpp
    # T04: strided_view tests
    unit/views/test_strided_view.cpp
    # T05: subview tests
    unit/views/test_subview.cpp
    # T06: global_iterator operator* fix
    unit/iterators/test_global_iterator.cpp
    # T07: Iterator integration
    unit/iterators/test_iterator_integration.cpp
)

# Phase 10: Futures, Remote, and RPC Fixes tests
target_sources(dtl_unit_tests PRIVATE
    # T01: Diagnostic collector future lifecycle wiring
    unit/futures/test_future_diagnostics.cpp
    # T07: algorithm_result.hpp unit tests
    unit/futures/test_algorithm_result.cpp
)

# Phase 13: Policy, Topology, and Serialization tests
target_sources(dtl_unit_tests PRIVATE
    unit/policies/test_policy_types.cpp
    unit/serialization/test_stl_serializers.cpp
    unit/serialization/test_aggregate_serializer.cpp
)

# Phase 09: Async Operation Completion tests
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_async_operations.cpp
)

# Phase 14: Test Coverage Expansion (Batch 1)
target_sources(dtl_unit_tests PRIVATE
    unit/memory/test_memory_coverage.cpp
    unit/containers/test_container_semantics.cpp
    unit/index/test_index_edge_cases.cpp
    unit/serialization/test_serialization_coverage.cpp
    unit/topology/test_topology_coverage.cpp
)

# Phase 14: Test Coverage Expansion (Batch 2)
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_backend_coverage.cpp
    unit/rma/test_rma_coverage.cpp
    unit/remote/test_remote_coverage.cpp
    unit/error/test_error_coverage.cpp
    unit/futures/test_futures_coverage.cpp
)

# Runtime subproject tests
target_sources(dtl_unit_tests PRIVATE
    unit/runtime/test_backend_discovery.cpp
    unit/runtime/test_plugin_loader.cpp
    unit/runtime/test_connection_pool.cpp
)

# Phase R7: Container Enhancements
target_sources(dtl_unit_tests PRIVATE
    unit/containers/test_distributed_map_batch.cpp
    unit/policies/test_block_nd_partition.cpp
    unit/views/test_composed_view.cpp
)

# Phase R10: Test Infrastructure Expansion
target_sources(dtl_unit_tests PRIVATE
    unit/error/test_error_recovery.cpp
    unit/stress/test_large_scale.cpp
    unit/stress/test_concurrent_access.cpp
)

# Phase 21: HIP Backend Parity & Device Abstraction Fixes (V1.5.1)
target_sources(dtl_unit_tests PRIVATE
    unit/hip/test_hip_device_buffer.cpp
    unit/hip/test_hip_algorithms.cpp
)

# Phase 22: Algorithm Stub Completion (V1.5.2)
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_rotate_distributed.cpp
    unit/algorithms/test_unique_distributed.cpp
    unit/algorithms/test_stable_sort_global_unit.cpp
    unit/algorithms/test_structural_mutation_conformance.cpp
)

# Phase 23: Runtime Stub Completion (V1.6.0)
# (plugin_loader and connection_pool tests already registered above)
# Test plugin DSOs for real dlopen testing
add_subdirectory(test_plugins)

# Phase 24: NCCL Point-to-Point & SHMEM Send/Recv (V1.6.1)
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_nccl_p2p.cpp
    unit/backend/test_shmem_p2p.cpp
)

# NCCL Comm Adapter (concept-compliant Layer 2)
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_nccl_comm_adapter.cpp
)

# Phase 25: Experimental Backend Completion (V1.7.0)
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_shared_region_posix.cpp
    unit/backend/test_shared_memory_comm.cpp
    unit/backend/test_hierarchical_communicator.cpp
)

# Work-stealing thread pool tests
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_work_stealing_pool.cpp
)

# Phase 27: Code Deduplication & Refactoring (V1.7.2)
target_sources(dtl_unit_tests PRIVATE
    unit/containers/test_policy_mixin.cpp
    unit/containers/test_common_ops.cpp
    unit/algorithms/test_dispatch_helper.cpp
)

# Phase 29: V2.0 Scaffolding - Observability stubs
target_sources(dtl_unit_tests PRIVATE
    unit/observe/test_observe_stubs.cpp
)

# Remediation 2026-02-13: Phase 1 Critical Fixes
target_sources(dtl_unit_tests PRIVATE
    unit/observe/test_metrics_include.cpp
    unit/backend/test_mpi_rma_lifecycle.cpp
)

# Remediation 2026-02-13: Phase 2 Algorithm Guards
target_sources(dtl_unit_tests PRIVATE
    unit/algorithms/test_min_element_guard.cpp
    unit/algorithms/test_minmax_element_guard.cpp
    unit/algorithms/test_nth_element_guard.cpp
    unit/algorithms/test_partition_guard.cpp
    unit/algorithms/test_partial_sort_guard.cpp
    unit/algorithms/test_partial_sort_copy_guard.cpp
    unit/algorithms/test_async_transform_reduce_guard.cpp
    unit/algorithms/test_adjacent_difference_guard.cpp
)

# Remediation 2026-02-13: Phase 3 Backend Error Hardening
target_sources(dtl_unit_tests PRIVATE
    unit/backend/test_shared_memory_invalid_region.cpp
)

# Remediation 2026-02-13: Phase 4 Policy/Infrastructure
target_sources(dtl_unit_tests PRIVATE
    unit/policies/test_consistency_sync_point.cpp
    unit/policies/test_prefetch_callable.cpp
    unit/backend/test_backend_context_lifecycle.cpp
    unit/backend/test_memory_space_compatibility.cpp
)

# Remediation 2026-02-13: Phase 5 V2 Scaffolding
target_sources(dtl_unit_tests PRIVATE
    unit/core/test_version_consistency.cpp
    unit/topology/test_rank_ordering_identity.cpp
)

# -----------------------------------------------------------------------------
# Test Discovery
# -----------------------------------------------------------------------------
gtest_discover_tests(dtl_unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    PROPERTIES
        LABELS "unit"
        TIMEOUT 60
)

# -----------------------------------------------------------------------------
# Create Test Directories (if they don't exist)
# -----------------------------------------------------------------------------
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/core)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/error)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/serialization)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/backend)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/utility)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/policies)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/index)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/views)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/containers)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/algorithms)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/futures)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/remote)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/topology)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/communication)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/rma)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/mpmd)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/memory)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/bindings)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/iterators)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/runtime)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/stress)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/unit/observe)

# -----------------------------------------------------------------------------
# Optional Serialization Adapter Test Configuration
# -----------------------------------------------------------------------------
# Tests for optional adapters are always compiled but use GTEST_SKIP()
# at runtime when dependencies are unavailable. This ensures compile-time
# correctness of the adapter code paths even without the external libraries.
#
# To run serialization adapter tests specifically:
#   ctest -L serialization
#
# When Cereal is available:
#   cmake -DDTL_ENABLE_CEREAL=ON ..
#   ctest --output-on-failure
#
# When Boost.Serialization is available:
#   cmake -DDTL_ENABLE_BOOST_SERIALIZATION=ON ..
#   ctest --output-on-failure

# -----------------------------------------------------------------------------
# Integration Tests (optional, MPI-dependent)
# -----------------------------------------------------------------------------
if(DTL_BUILD_INTEGRATION_TESTS AND DTL_MPI_AVAILABLE)
    # MPI Communicator Integration Tests
    # Note: These tests have their own main() for MPI initialization
    add_executable(dtl_mpi_tests
        integration/test_mpi_communicator.cpp
        integration/test_algorithms_mpi.cpp
        integration/test_rma_mpi.cpp
        integration/test_mpi_sort.cpp
        integration/test_scan_mpi.cpp
        integration/test_context_mpi.cpp
        integration/test_connection_pool_mpi.cpp
        integration/test_distributed_vector_mpi.cpp
        mpi/remote/test_async_remote_ref.cpp
        mpi/remote/test_dynamic_handler_remote.cpp
        mpi/test_stable_sort_global.cpp
    )

    target_compile_features(dtl_mpi_tests PRIVATE cxx_std_20)

    target_link_libraries(dtl_mpi_tests PRIVATE
        DTL::dtl
        DTL::mpi
        DTL::cpu
        MPI::MPI_CXX
        GTest::gtest
    )

    # Apply compiler warnings
    dtl_set_compiler_warnings(dtl_mpi_tests)

    # Create integration test directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/integration)

    # MPI tests require mpirun - add as manual test
    # Run with: mpirun -np 2 ./dtl_mpi_tests
    # or:       mpirun -np 4 ./dtl_mpi_tests
    add_test(NAME mpi_integration_2ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dtl_mpi_tests> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(mpi_integration_2ranks PROPERTIES
        LABELS "integration;mpi"
        TIMEOUT 120
    )

    add_test(NAME mpi_integration_4ranks
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS} $<TARGET_FILE:dtl_mpi_tests> ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(mpi_integration_4ranks PROPERTIES
        LABELS "integration;mpi"
        TIMEOUT 120
    )
endif()

# -----------------------------------------------------------------------------
# CUDA Integration Tests
# -----------------------------------------------------------------------------
if(DTL_BUILD_INTEGRATION_TESTS AND DTL_ENABLE_CUDA)
    # Enable CUDA language for .cu files
    enable_language(CUDA)

    add_executable(dtl_cuda_tests
        integration/test_cuda_memory_space.cpp
        integration/test_cuda_algorithms.cu
    )

    target_compile_features(dtl_cuda_tests PRIVATE cxx_std_20)

    # Set CUDA standard
    set_target_properties(dtl_cuda_tests PROPERTIES CUDA_STANDARD 20)

    target_link_libraries(dtl_cuda_tests PRIVATE
        DTL::dtl
        DTL::cuda
        CUDA::cudart
        GTest::gtest
        GTest::gtest_main
    )

    # Apply compiler warnings
    dtl_set_compiler_warnings(dtl_cuda_tests)

    # Add as CTest
    gtest_discover_tests(dtl_cuda_tests
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTIES
            LABELS "integration;cuda"
            TIMEOUT 120
    )
endif()

# -----------------------------------------------------------------------------
# SHMEM Integration Tests
# -----------------------------------------------------------------------------
if(DTL_BUILD_INTEGRATION_TESTS AND DTL_ENABLE_SHMEM)
    add_executable(dtl_shmem_tests
        integration/test_shmem_backend.cpp
    )

    target_compile_features(dtl_shmem_tests PRIVATE cxx_std_20)

    target_link_libraries(dtl_shmem_tests PRIVATE
        DTL::dtl
        DTL::shmem
        GTest::gtest
        GTest::gtest_main
    )

    # Apply compiler warnings
    dtl_set_compiler_warnings(dtl_shmem_tests)

    # SHMEM tests require oshrun - add as manual test
    # Run with: oshrun -np 2 ./dtl_shmem_tests
    # or:       oshrun -np 4 ./dtl_shmem_tests
    if(SHMEM_OSHRUN_EXECUTABLE)
        add_test(NAME shmem_integration_2pes
            COMMAND ${SHMEM_OSHRUN_EXECUTABLE} -np 2 $<TARGET_FILE:dtl_shmem_tests>
        )
        set_tests_properties(shmem_integration_2pes PROPERTIES
            LABELS "integration;shmem"
            TIMEOUT 120
        )

        add_test(NAME shmem_integration_4pes
            COMMAND ${SHMEM_OSHRUN_EXECUTABLE} -np 4 $<TARGET_FILE:dtl_shmem_tests>
        )
        set_tests_properties(shmem_integration_4pes PROPERTIES
            LABELS "integration;shmem"
            TIMEOUT 120
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# NCCL Integration Tests
# -----------------------------------------------------------------------------
if(DTL_BUILD_INTEGRATION_TESTS AND DTL_ENABLE_NCCL AND DTL_ENABLE_CUDA AND DTL_MPI_AVAILABLE)
    add_subdirectory(mpi/nccl)
endif()

# -----------------------------------------------------------------------------
# Multi-Backend Integration Tests (Phase 18)
# -----------------------------------------------------------------------------
if(DTL_BUILD_INTEGRATION_TESTS)
    add_subdirectory(integration)
endif()

# -----------------------------------------------------------------------------
# Custom Test Target (for convenience)
# -----------------------------------------------------------------------------
add_custom_target(run_unit_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -L unit --output-on-failure
    DEPENDS dtl_unit_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running DTL unit tests..."
)

# -----------------------------------------------------------------------------
# C Bindings Tests
# -----------------------------------------------------------------------------
if(DTL_BUILD_C_BINDINGS)
    add_subdirectory(bindings/c)
endif()

# -----------------------------------------------------------------------------
# Fuzz Targets (optional, requires Clang)
# -----------------------------------------------------------------------------
if(DTL_BUILD_FUZZ_TARGETS)
    add_subdirectory(fuzz)
endif()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "DTL Tests Configuration:")
message(STATUS "  Unit tests target: dtl_unit_tests")
if(TARGET dtl_mpi_tests OR TARGET dtl_cuda_tests OR TARGET dtl_shmem_tests)
    message(STATUS "  Integration tests enabled: ${DTL_BUILD_INTEGRATION_TESTS}")
endif()
if(DTL_BUILD_C_BINDINGS)
    message(STATUS "  C bindings tests target: test_c_bindings")
endif()
