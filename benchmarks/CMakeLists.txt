# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause

# Benchmark build configuration
# Requires Google Benchmark (https://github.com/google/benchmark)

find_package(benchmark REQUIRED)

# Helper function to create a benchmark executable
function(dtl_add_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE
        DTL::dtl
        benchmark::benchmark
        benchmark::benchmark_main
    )
    target_include_directories(${name} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    target_compile_features(${name} PRIVATE cxx_std_20)
    # Suppress deprecation warnings from legacy constructors used in benchmarks
    target_compile_definitions(${name} PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)
endfunction()

# Algorithm benchmarks
dtl_add_benchmark(bench_algorithms_reduce   algorithms/bench_reduce.cpp)
dtl_add_benchmark(bench_algorithms_sort     algorithms/bench_sort.cpp)
dtl_add_benchmark(bench_algorithms_transform algorithms/bench_transform.cpp)
dtl_add_benchmark(bench_algorithms_scan     algorithms/bench_scan.cpp)

# Communication benchmarks (overhead only, no actual MPI)
dtl_add_benchmark(bench_comm_collectives    communication/bench_collectives.cpp)
dtl_add_benchmark(bench_comm_halo_exchange  communication/bench_halo_exchange.cpp)
dtl_add_benchmark(bench_comm_point_to_point communication/bench_point_to_point.cpp)

# Container benchmarks
dtl_add_benchmark(bench_containers_vector   containers/bench_vector_access.cpp)
dtl_add_benchmark(bench_containers_view     containers/bench_view_operations.cpp)

# Memory benchmarks
dtl_add_benchmark(bench_memory_allocation   memory/bench_allocation.cpp)
dtl_add_benchmark(bench_memory_transfer     memory/bench_memory_transfer.cpp)

# Futures benchmarks
dtl_add_benchmark(bench_futures             futures/bench_futures.cpp)

# RMA benchmarks
dtl_add_benchmark(bench_rma                 rma/bench_rma.cpp)

# Backend benchmarks
dtl_add_benchmark(bench_cpu_executor        baselines/bench_cpu_executor.cpp)

# Backend-specific benchmarks (MPI, CUDA, NCCL)
if(DTL_MPI_AVAILABLE)
    dtl_add_benchmark(bench_mpi_collectives backend/bench_mpi_collectives.cpp)
    target_link_libraries(bench_mpi_collectives PRIVATE MPI::MPI_CXX)
endif()

if(DTL_ENABLE_CUDA)
    dtl_add_benchmark(bench_cuda_operations backend/bench_cuda_operations.cpp)
    target_link_libraries(bench_cuda_operations PRIVATE CUDA::cudart)
    if(TARGET DTL::cuda)
        target_link_libraries(bench_cuda_operations PRIVATE DTL::cuda)
    endif()
endif()

if(DTL_ENABLE_NCCL AND DTL_ENABLE_CUDA)
    dtl_add_benchmark(bench_nccl_collectives backend/bench_nccl_collectives.cpp)
    target_link_libraries(bench_nccl_collectives PRIVATE CUDA::cudart)
    if(TARGET DTL::cuda)
        target_link_libraries(bench_nccl_collectives PRIVATE DTL::cuda)
    endif()
    if(TARGET DTL::nccl)
        target_link_libraries(bench_nccl_collectives PRIVATE DTL::nccl)
    endif()
    if(DTL_MPI_AVAILABLE)
        target_link_libraries(bench_nccl_collectives PRIVATE MPI::MPI_CXX)
    endif()
endif()

# C binding benchmarks
if(DTL_BUILD_C_BINDINGS)
    dtl_add_benchmark(bench_c_api bindings/bench_c_api.cpp)
    target_link_libraries(bench_c_api PRIVATE dtl_c)
endif()

# Scaling benchmarks (MPI multi-rank strong/weak scaling)
if(DTL_MPI_AVAILABLE)
    add_executable(bench_scaling scaling/bench_scaling.cpp)
    target_link_libraries(bench_scaling PRIVATE
        DTL::dtl
        DTL::cpu
        MPI::MPI_CXX
        benchmark::benchmark
    )
    target_include_directories(bench_scaling PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/backends
    )
    target_compile_features(bench_scaling PRIVATE cxx_std_20)
    target_compile_definitions(bench_scaling PRIVATE DTL_SUPPRESS_DEPRECATION_WARNINGS=1)
endif()
