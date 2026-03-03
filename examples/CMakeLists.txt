# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Examples build configuration
# =============================================================================
# DTL example programs demonstrating library usage
# Builds 64 examples: 29 C++, 13 C, 12 Python (scripts), 10 Fortran
# =============================================================================

# -----------------------------------------------------------------------------
# Build Macros
# -----------------------------------------------------------------------------

# C++ MPI example (requires DTL_MPI_AVAILABLE)
macro(dtl_add_mpi_example name source)
    add_executable(${name} ${source})
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_link_libraries(${name} PRIVATE
        DTL::dtl
        DTL::mpi
        DTL::cpu
        MPI::MPI_CXX
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
endmacro()

# C++ non-MPI example (always available)
macro(dtl_add_example name source)
    add_executable(${name} ${source})
    target_compile_features(${name} PRIVATE cxx_std_20)
    target_link_libraries(${name} PRIVATE
        DTL::dtl
        DTL::cpu
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -Wall -Wextra)
    endif()
endmacro()

# C example (requires DTL_BUILD_C_BINDINGS)
macro(dtl_add_c_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE dtl_c)
    if(DTL_MPI_AVAILABLE AND TARGET MPI::MPI_C)
        target_link_libraries(${name} PRIVATE MPI::MPI_C)
    endif()
endmacro()

# =============================================================================
# C++ Non-MPI Examples (always available)
# =============================================================================

# Basics
dtl_add_example(hello_distributed basics/hello_distributed.cpp)

message(STATUS "DTL Examples: hello_distributed enabled")

# =============================================================================
# C++ MPI Examples
# =============================================================================

if(DTL_MPI_AVAILABLE)
    # --- Existing ---
    dtl_add_mpi_example(parallel_reduce         algorithms/parallel_reduce.cpp)
    dtl_add_mpi_example(distributed_vector_sum  basics/distributed_vector_sum.cpp)
    dtl_add_mpi_example(local_view_stl          basics/local_view_stl.cpp)
    dtl_add_mpi_example(environment_context     basics/environment_context.cpp)
    dtl_add_mpi_example(transform_reduce        algorithms/transform_reduce.cpp)
    dtl_add_mpi_example(distributed_sort        algorithms/distributed_sort.cpp)
    dtl_add_mpi_example(custom_partition        advanced/custom_partition.cpp)
    dtl_add_mpi_example(hybrid_backend          advanced/hybrid_backend.cpp)
    dtl_add_mpi_example(halo_exchange           advanced/halo_exchange.cpp)
    dtl_add_mpi_example(async_pipeline          async/async_pipeline.cpp)
    dtl_add_mpi_example(continuation_chaining   async/continuation_chaining.cpp)
    dtl_add_mpi_example(coordinator_worker      mpmd/coordinator_worker.cpp)
    dtl_add_mpi_example(heterogeneous_roles     mpmd/heterogeneous_roles.cpp)

    # --- New: Basics ---
    dtl_add_mpi_example(vector_allreduce_dtl    basics/vector_allreduce_dtl.cpp)
    dtl_add_mpi_example(collective_ops_dtl      basics/collective_ops_dtl.cpp)
    dtl_add_mpi_example(point_to_point_dtl      basics/point_to_point_dtl.cpp)

    # --- New: Algorithms ---
    dtl_add_mpi_example(scan_prefix             algorithms/scan_prefix.cpp)
    dtl_add_mpi_example(distributed_find_count  algorithms/distributed_find_count.cpp)

    # --- New: RMA ---
    dtl_add_mpi_example(rma_put_get             rma/rma_put_get.cpp)
    dtl_add_mpi_example(rma_passive_target      rma/rma_passive_target.cpp)
    dtl_add_mpi_example(rma_atomics             rma/rma_atomics.cpp)

    # --- New: Advanced ---
    dtl_add_mpi_example(topology_discovery      advanced/topology_discovery.cpp)
    dtl_add_mpi_example(distributed_tensor_ops  advanced/distributed_tensor_ops.cpp)

    # --- New: MPMD ---
    dtl_add_mpi_example(role_manager_dtl        mpmd/role_manager_dtl.cpp)

    # --- New: Applications ---
    dtl_add_mpi_example(monte_carlo_pi          applications/monte_carlo_pi.cpp)
    dtl_add_mpi_example(jacobi_1d               applications/jacobi_1d.cpp)

    # --- New: Async ---
    dtl_add_mpi_example(futures_dtl             async/futures_dtl.cpp)

    # --- New: Views ---
    dtl_add_mpi_example(segmented_view_example  views/segmented_view_example.cpp)
    dtl_add_mpi_example(strided_view_example    views/strided_view_example.cpp)

    # --- New: Algorithms ---
    dtl_add_mpi_example(stable_sort_example     algorithms/stable_sort_example.cpp)

    # --- New: Containers ---
    dtl_add_mpi_example(distributed_map_example       containers/distributed_map_example.cpp)
    dtl_add_mpi_example(distributed_array_example     containers/distributed_array_example.cpp)
    dtl_add_mpi_example(distributed_span_example      containers/distributed_span_example.cpp)

    # --- Mini-apps: multi-feature HPC workloads ---
    add_subdirectory(mini-apps)

    message(STATUS "DTL Examples: C++ MPI examples enabled (14 existing + 22 new)")
endif()

# =============================================================================
# C++ GPU Examples (require DTL_ENABLE_CUDA)
# =============================================================================

if(DTL_ENABLE_CUDA)
    # Enable CUDA language for GPU examples that use kernel syntax
    enable_language(CUDA)

    # GPU examples use CUDA kernel syntax, so they must be compiled with nvcc (.cu files)
    add_executable(gpu_accelerated_transform gpu/gpu_accelerated_transform.cu)
    set_target_properties(gpu_accelerated_transform PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(gpu_accelerated_transform PRIVATE
        DTL::dtl DTL::cpu CUDA::cudart
    )

    add_executable(gpu_resident_vector gpu/gpu_resident_vector.cu)
    set_target_properties(gpu_resident_vector PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(gpu_resident_vector PRIVATE
        DTL::dtl DTL::cpu CUDA::cudart
    )

    if(DTL_MPI_AVAILABLE)
        target_link_libraries(gpu_accelerated_transform PRIVATE DTL::mpi MPI::MPI_CXX)
        target_link_libraries(gpu_resident_vector PRIVATE DTL::mpi MPI::MPI_CXX)
    endif()

    message(STATUS "DTL Examples: GPU examples enabled")
endif()

# =============================================================================
# C Examples (require DTL_BUILD_C_BINDINGS)
# =============================================================================

if(DTL_BUILD_C_BINDINGS)
    # Enable C language for C examples
    enable_language(C)

    # Find MPI C component if MPI is available
    if(DTL_MPI_AVAILABLE AND NOT TARGET MPI::MPI_C)
        find_package(MPI COMPONENTS C QUIET)
    endif()

    # --- Existing ---
    dtl_add_c_example(c_hello_dtl            c/hello_dtl.c)
    dtl_add_c_example(c_distributed_vector   c/distributed_vector.c)
    dtl_add_c_example(c_collective_ops       c/collective_ops.c)
    dtl_add_c_example(c_distributed_tensor   c/distributed_tensor.c)

    # --- New ---
    dtl_add_c_example(c_point_to_point       c/point_to_point.c)
    dtl_add_c_example(c_rma_operations       c/rma_operations.c)
    dtl_add_c_example(c_topology_query       c/topology_query.c)
    dtl_add_c_example(c_mpmd_roles           c/mpmd_roles.c)
    dtl_add_c_example(c_algorithms           c/algorithms.c)
    dtl_add_c_example(c_scan_prefix          c/scan_prefix.c)
    dtl_add_c_example(c_futures_rpc          c/futures_rpc.c)
    dtl_add_c_example(c_monte_carlo_pi      c/monte_carlo_pi.c)
    dtl_add_c_example(c_jacobi_1d           c/jacobi_1d.c)

    message(STATUS "DTL Examples: C examples enabled (4 existing + 9 new)")
endif()

# =============================================================================
# SHMEM Examples
# =============================================================================

if(DTL_ENABLE_SHMEM)
    # Macro for SHMEM examples
    macro(dtl_add_shmem_example name source)
        add_executable(${name} ${source})
        target_compile_features(${name} PRIVATE cxx_std_20)
        target_link_libraries(${name} PRIVATE
            DTL::dtl
            DTL::shmem
        )
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${name} PRIVATE -Wall -Wextra)
        endif()
    endmacro()

    dtl_add_shmem_example(shmem_put_get_demo    shmem/shmem_put_get_demo.cpp)
    dtl_add_shmem_example(shmem_atomics_demo    shmem/shmem_atomics_demo.cpp)

    message(STATUS "DTL Examples: SHMEM examples enabled (2 examples)")
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "DTL Examples Configuration:")
message(STATUS "  C++ non-MPI examples: ON")
message(STATUS "  C++ MPI examples:     ${DTL_MPI_AVAILABLE}")
message(STATUS "  C++ GPU examples:     ${DTL_ENABLE_CUDA}")
message(STATUS "  C++ SHMEM examples:   ${DTL_ENABLE_SHMEM}")
message(STATUS "  C examples:           ${DTL_BUILD_C_BINDINGS}")
message(STATUS "  Fortran examples:     ${DTL_BUILD_FORTRAN} (via bindings/fortran/examples)")
if(DTL_BUILD_C_BINDINGS AND CMAKE_Fortran_COMPILER)
    message(STATUS "  Fortran examples:     ON")
else()
    message(STATUS "  Fortran examples:     OFF")
endif()
