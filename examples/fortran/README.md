# DTL Fortran Examples

This directory contains legacy Fortran example sources kept for reference only.
The supported Fortran examples for `v0.1.0-alpha.1` are built from
`bindings/fortran/examples`, which link against the canonical `dtl_fortran`
target and consume the real `dtl` module.

## Prerequisites

Build DTL with C and Fortran bindings:
```bash
cmake -B build \
    -DDTL_BUILD_C_BINDINGS=ON \
    -DDTL_BUILD_FORTRAN=ON \
    -DDTL_BUILD_EXAMPLES=ON
cmake --build build
```

## Files

- `hello_dtl.f90` - Basic example showing context creation
- `distributed_vector.f90` - Distributed vector operations

## Building Examples

Supported examples are built automatically from `bindings/fortran/examples`
when `DTL_BUILD_EXAMPLES=ON` and `DTL_BUILD_FORTRAN=ON`:

```bash
cmake -B build \
    -DDTL_BUILD_C_BINDINGS=ON \
    -DDTL_BUILD_FORTRAN=ON
cmake --build build
```

### CMake Consumer Integration

For downstream projects using `find_package(DTL)`:

```cmake
find_package(DTL REQUIRED)
add_executable(my_fortran_app my_app.f90)
target_link_libraries(my_fortran_app PRIVATE DTL::dtl_fortran)
```

## Running Examples

```bash
# Single process
./build/bin/fortran/dtl_fortran_hello
./build/bin/fortran/dtl_fortran_vector_demo

# MPI parallel
mpirun -np 4 ./build/bin/fortran/dtl_fortran_hello
mpirun -np 4 ./build/bin/fortran/dtl_fortran_vector_demo
```

These legacy example sources are intentionally not wired into the top-level
`examples` target to keep host-only and non-Fortran builds working by default.

## Module API

The `dtl` module provides:

### Constants

```fortran
! Data types
integer, parameter :: DTL_DTYPE_FLOAT64 = 9
integer, parameter :: DTL_DTYPE_FLOAT32 = 8
integer, parameter :: DTL_DTYPE_INT64   = 3
integer, parameter :: DTL_DTYPE_INT32   = 2

! Reduce operations
integer, parameter :: DTL_OP_SUM  = 0
integer, parameter :: DTL_OP_PROD = 1
integer, parameter :: DTL_OP_MIN  = 2
integer, parameter :: DTL_OP_MAX  = 3
```

### Context Management

```fortran
function dtl_context_create_default(ctx) result(status)
subroutine dtl_context_destroy(ctx)
function dtl_context_rank(ctx) result(rank)
function dtl_context_size(ctx) result(size)
function dtl_context_is_root(ctx) result(is_root)
```

### Distributed Vector

```fortran
function dtl_vector_create(ctx, dtype, size, vec) result(status)
subroutine dtl_vector_destroy(vec)
function dtl_vector_global_size(vec) result(size)
function dtl_vector_local_size(vec) result(size)
function dtl_vector_local_offset(vec) result(offset)
function dtl_vector_local_data_mut(vec) result(ptr)
function dtl_vector_fill(vec, value) result(status)
```

### Collective Operations

```fortran
function dtl_barrier(ctx) result(status)
function dtl_broadcast(ctx, data, count, dtype, root) result(status)
function dtl_reduce(ctx, sendbuf, recvbuf, count, dtype, op, root) result(status)
function dtl_allreduce(ctx, sendbuf, recvbuf, count, dtype, op) result(status)
```

### Helper Functions

```fortran
function shape_1d(n1) result(s)
function shape_2d(n1, n2) result(s)
function shape_3d(n1, n2, n3) result(s)
function is_success(status) result(ok)
```

## Working with Data Pointers

The C bindings return `type(c_ptr)` pointers. Convert to Fortran array pointers using:

```fortran
use, intrinsic :: iso_c_binding

type(c_ptr) :: data_ptr
real(c_double), pointer :: data(:)
integer(c_int64_t) :: n

! Get the C pointer
data_ptr = dtl_vector_local_data_mut(vec)

! Convert to Fortran pointer with size n
call c_f_pointer(data_ptr, data, [n])

! Now use data(:) as a normal Fortran array
data(:) = 42.0_c_double
```

## Error Handling

All functions return a status code. Check with:

```fortran
integer(c_int32_t) :: status

status = dtl_vector_create(ctx, dtype, size, vec)
if (.not. is_success(status)) then
    print *, 'Error!'
    stop 1
end if
```

## Legacy Manual Build

For environments without CMake, you can compile manually:

```bash
gfortran -c dtl.f90
gfortran -L../../build/src/bindings/c hello_dtl.f90 dtl.o -ldtl_c -o hello_dtl
```

Note: The CMake-based build path above is the recommended approach.
