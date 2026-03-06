# Fortran Bindings Guide

This guide covers using DTL from Fortran through the native `dtl` module, which
wraps the C API via Fortran 2003's `ISO_C_BINDING`.

For `v0.1.0-alpha.1`, the supported Fortran examples are built from
`bindings/fortran/examples`. The legacy `examples/fortran` directory is
reference-only and is not part of the default top-level example build.

---

## Table of Contents

- [Overview](#overview)
- [Distributed Span in Fortran](#distributed-span-in-fortran)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Building with CMake](#building-with-cmake)
- [Native Module Usage](#native-module-usage)
- [Context Operations](#context-operations)
- [Container Operations](#container-operations)
- [Error Handling](#error-handling)
- [Complete Examples](#complete-examples)
- [Manual Interface (Legacy)](#manual-interface-legacy)
- [Building and Running](#building-and-running)

---

## Overview

DTL provides native Fortran bindings through the `dtl` module. This provides:

- **Zero Manual Interfaces**: Just `use dtl` - no interface blocks needed
- **Full C API Access**: All core C API functions wrapped
- **Direct Memory Access**: Work with Fortran arrays pointing to DTL data
- **MPI Integration**: Works seamlessly with MPI Fortran bindings
- **Type Safety**: Fortran's strong typing with explicit C interop

### Distributed Span in Fortran

The Fortran module exposes first-class span bindings through the C ABI:

- creation: `dtl_span_from_vector`, `dtl_span_from_array`, `dtl_span_from_tensor`, `dtl_span_create`
- lifecycle: `dtl_span_destroy`, `dtl_span_is_valid`
- metadata: `dtl_span_size`, `dtl_span_local_size`, `dtl_span_rank`, `dtl_span_num_ranks`
- local data and subspans: `dtl_span_data_mut`, `dtl_span_first`, `dtl_span_last`, `dtl_span_subspan`

As in C/C++, spans are non-owning. The backing container handle must outlive all derived span handles and any `c_f_pointer` projections.

---

## Quick Start

```fortran
program hello_dtl
    use dtl
    implicit none
    
    type(c_ptr) :: ctx
    integer(c_int) :: status
    
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) stop 'Failed'
    
    print *, 'Rank', dtl_context_rank(ctx), 'of', dtl_context_size(ctx)
    
    call dtl_context_destroy(ctx)
end program
```

---

## Prerequisites

- Fortran 2003+ compiler with `ISO_C_BINDING` support:
  - gfortran 4.3+
  - ifort 10+
  - flang
  - nvfortran
- DTL C bindings library (`libdtl_c.so` or `libdtl_c.a`)
- DTL Fortran module (`libdtl_fortran.a` + `dtl.mod`)
- MPI Fortran bindings (optional, for multi-rank execution)

---

## Building with CMake

### Enable Fortran Bindings

```bash
cmake -B build \
    -DDTL_BUILD_C_BINDINGS=ON \
    -DDTL_BUILD_FORTRAN=ON \
    -DDTL_BUILD_EXAMPLES=ON

cmake --build build
```

### Requirements

- `DTL_BUILD_FORTRAN=ON` requires `DTL_BUILD_C_BINDINGS=ON`
- A Fortran compiler must be available
- CMake will automatically detect and enable the Fortran language

### Build Outputs

- `lib/libdtl_fortran.a` - Fortran module library
- `include/dtl/fortran/dtl.mod` - Fortran module file
- `bin/fortran/dtl_fortran_hello` - Example program
- `bin/fortran/dtl_fortran_vector_demo` - Vector example program

---

## Native Module Usage

### Import the Module

```fortran
program my_program
    use dtl
    implicit none
    ! Your code here
end program
```

### Available Constants

#### Status Codes
```fortran
DTL_SUCCESS                 ! Operation succeeded
DTL_ERROR_COMMUNICATION     ! Communication error
DTL_ERROR_MEMORY            ! Memory allocation error
DTL_ERROR_BOUNDS            ! Index out of bounds
DTL_ERROR_INVALID_ARGUMENT  ! Invalid argument
```

#### Data Types
```fortran
DTL_DTYPE_INT32    ! 32-bit integer
DTL_DTYPE_INT64    ! 64-bit integer
DTL_DTYPE_FLOAT32  ! Single precision float
DTL_DTYPE_FLOAT64  ! Double precision float
```

#### Reduction Operations
```fortran
DTL_OP_SUM   ! Sum
DTL_OP_PROD  ! Product
DTL_OP_MIN   ! Minimum
DTL_OP_MAX   ! Maximum
```

---

## Manual Interface (Legacy)

> **Note**: This section documents the legacy approach of manually writing interface blocks.
> The native `dtl` module is now preferred for new projects.

If you need to create custom interfaces or extend the bindings, you can create a Fortran module that declares DTL's C interface:

### dtl_bindings.f90

```fortran
!> DTL Fortran Interface Module (Legacy/Custom)
!> Provides ISO_C_BINDING declarations for DTL C API
module dtl_bindings
    use, intrinsic :: iso_c_binding
    implicit none

    ! Opaque handle types
    type, bind(c) :: dtl_context_handle
        type(c_ptr) :: ptr = c_null_ptr
    end type

    type, bind(c) :: dtl_vector_handle
        type(c_ptr) :: ptr = c_null_ptr
    end type

    type, bind(c) :: dtl_tensor_handle
        type(c_ptr) :: ptr = c_null_ptr
    end type

    ! Status codes
    integer(c_int), parameter :: DTL_SUCCESS = 0
    integer(c_int), parameter :: DTL_ERROR_COMMUNICATION = 100
    integer(c_int), parameter :: DTL_ERROR_MEMORY = 200
    integer(c_int), parameter :: DTL_ERROR_BOUNDS = 400
    integer(c_int), parameter :: DTL_ERROR_BACKEND = 500

    ! Data types
    integer(c_int), parameter :: DTL_DTYPE_INT32 = 2
    integer(c_int), parameter :: DTL_DTYPE_INT64 = 3
    integer(c_int), parameter :: DTL_DTYPE_FLOAT32 = 8
    integer(c_int), parameter :: DTL_DTYPE_FLOAT64 = 9

    ! Reduction operations
    integer(c_int), parameter :: DTL_OP_SUM = 0
    integer(c_int), parameter :: DTL_OP_PROD = 1
    integer(c_int), parameter :: DTL_OP_MIN = 2
    integer(c_int), parameter :: DTL_OP_MAX = 3

    ! Interface declarations
    interface

        !> Create default context (MPI_COMM_WORLD)
        function dtl_context_create_default(ctx) bind(c, name='dtl_context_create_default')
            import :: c_ptr, c_int
            type(c_ptr), intent(out) :: ctx
            integer(c_int) :: dtl_context_create_default
        end function

        !> Destroy context
        subroutine dtl_context_destroy(ctx) bind(c, name='dtl_context_destroy')
            import :: c_ptr
            type(c_ptr), value :: ctx
        end subroutine

        !> Get rank
        function dtl_context_rank(ctx) bind(c, name='dtl_context_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_rank
        end function

        !> Get size
        function dtl_context_size(ctx) bind(c, name='dtl_context_size')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_size
        end function

        !> Barrier synchronization
        function dtl_context_barrier(ctx) bind(c, name='dtl_context_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: ctx
            integer(c_int) :: dtl_context_barrier
        end function

        !> Create vector
        function dtl_vector_create(ctx, dtype, global_size, vec) &
                bind(c, name='dtl_vector_create')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: global_size
            type(c_ptr), intent(out) :: vec
            integer(c_int) :: dtl_vector_create
        end function

        !> Destroy vector
        subroutine dtl_vector_destroy(vec) bind(c, name='dtl_vector_destroy')
            import :: c_ptr
            type(c_ptr), value :: vec
        end subroutine

        !> Get global size
        function dtl_vector_global_size(vec) bind(c, name='dtl_vector_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_global_size
        end function

        !> Get local size
        function dtl_vector_local_size(vec) bind(c, name='dtl_vector_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_size
        end function

        !> Get local offset
        function dtl_vector_local_offset(vec) bind(c, name='dtl_vector_local_offset')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_offset
        end function

        !> Get mutable local data pointer
        function dtl_vector_local_data_mut(vec) bind(c, name='dtl_vector_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: vec
            type(c_ptr) :: dtl_vector_local_data_mut
        end function

        !> Allreduce
        function dtl_allreduce(ctx, sendbuf, recvbuf, count, dtype, op) &
                bind(c, name='dtl_allreduce')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: sendbuf
            type(c_ptr), value :: recvbuf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: op
            integer(c_int) :: dtl_allreduce
        end function

        !> Broadcast
        function dtl_broadcast(ctx, buf, count, dtype, root) &
                bind(c, name='dtl_broadcast')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            type(c_ptr), value :: buf
            integer(c_int64_t), value :: count
            integer(c_int), value :: dtype
            integer(c_int), value :: root
            integer(c_int) :: dtl_broadcast
        end function

        !> Get error message
        function dtl_status_message(status) bind(c, name='dtl_status_message')
            import :: c_ptr, c_int
            integer(c_int), value :: status
            type(c_ptr) :: dtl_status_message
        end function

        !> Check MPI availability
        function dtl_has_mpi() bind(c, name='dtl_has_mpi')
            import :: c_int
            integer(c_int) :: dtl_has_mpi
        end function

    end interface

contains

    !> Convert C string pointer to Fortran string
    function c_to_f_string(c_str) result(f_str)
        type(c_ptr), intent(in) :: c_str
        character(len=:), allocatable :: f_str
        character(kind=c_char), pointer :: chars(:)
        integer :: i, length

        if (.not. c_associated(c_str)) then
            f_str = ""
            return
        end if

        ! Find string length
        length = 0
        call c_f_pointer(c_str, chars, [1000])
        do i = 1, 1000
            if (chars(i) == c_null_char) exit
            length = length + 1
        end do

        allocate(character(len=length) :: f_str)
        do i = 1, length
            f_str(i:i) = chars(i)
        end do
    end function

    !> Get error message as Fortran string
    function get_error_message(status) result(msg)
        integer(c_int), intent(in) :: status
        character(len=:), allocatable :: msg
        msg = c_to_f_string(dtl_status_message(status))
    end function

end module dtl_bindings
```

---

## Basic Usage

### Hello World

```fortran
program dtl_hello
    use dtl_bindings
    implicit none

    type(c_ptr) :: ctx
    integer(c_int) :: status, rank, size

    ! Create context
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) then
        print *, 'Error creating context: ', get_error_message(status)
        stop 1
    end if

    ! Get rank and size
    rank = dtl_context_rank(ctx)
    size = dtl_context_size(ctx)

    print '(A,I0,A,I0)', 'Rank ', rank, ' of ', size

    ! Barrier
    status = dtl_context_barrier(ctx)

    ! Cleanup
    call dtl_context_destroy(ctx)

end program dtl_hello
```

---

## Environment Operations

The environment manages backend lifecycle. Use environment factory methods to create contexts (preferred over `dtl_context_create_default`):

```fortran
type(c_ptr) :: env, ctx
integer(c_int) :: status

! Create environment (initializes MPI and other backends)
status = dtl_environment_create(env)
if (status /= DTL_SUCCESS) stop 'Environment creation failed'

! Create world context from environment
status = dtl_environment_make_world_context(env, ctx)
if (status /= DTL_SUCCESS) stop 'Context creation failed'

print *, 'Rank', dtl_context_rank(ctx), 'of', dtl_context_size(ctx)

! Cleanup
call dtl_context_destroy(ctx)
call dtl_environment_destroy(env)
```

The Fortran interface blocks for environment functions follow the same `ISO_C_BINDING` pattern as context operations. See the C API reference (`dtl_environment.h`) for the full list of available functions.

---

## Context Operations

Contexts can be created directly or via environment factory methods (preferred).

```fortran
type(c_ptr) :: ctx
integer(c_int) :: status, rank, size

! Create context
status = dtl_context_create_default(ctx)
if (status /= DTL_SUCCESS) stop 'Context creation failed'

! Query properties
rank = dtl_context_rank(ctx)
size = dtl_context_size(ctx)

! Synchronize
status = dtl_context_barrier(ctx)

! Cleanup (MUST call when done)
call dtl_context_destroy(ctx)
```

### NCCL Mode-Aware Context APIs (Fortran via C Interop)

When CUDA/NCCL are available, Fortran bindings expose mode-aware NCCL context
composition and capability queries:

```fortran
integer(c_int) :: status, mode, can_native, can_hybrid
type(c_ptr) :: base_ctx, nccl_ctx, split_ctx

status = dtl_context_with_nccl_ex(base_ctx, 0_c_int, DTL_NCCL_MODE_NATIVE_ONLY, nccl_ctx)
status = dtl_context_split_nccl_ex(nccl_ctx, 0_c_int, 0_c_int, 0_c_int, &
                                   DTL_NCCL_MODE_HYBRID_PARITY, split_ctx)

mode = dtl_context_nccl_mode(split_ctx)
can_native = dtl_context_nccl_supports_native(split_ctx, DTL_NCCL_OP_ALLREDUCE)
can_hybrid = dtl_context_nccl_supports_hybrid(split_ctx, DTL_NCCL_OP_SCAN)
```

Fortran also exposes explicit C-interoperable NCCL device collective procedures
such as:

- `dtl_nccl_allreduce_device(_ex)`
- `dtl_nccl_broadcast_device(_ex)`
- `dtl_nccl_scan_device_ex` / `dtl_nccl_exscan_device_ex`
- variable-size parity families: `dtl_nccl_*v_device_ex`

---

## Container Operations

### Working with Vectors

```fortran
use dtl_bindings
implicit none

type(c_ptr) :: ctx, vec
integer(c_int) :: status
integer(c_int64_t) :: global_size, local_size, local_offset, i
real(c_double), pointer :: data(:)
type(c_ptr) :: data_ptr

! Create context
status = dtl_context_create_default(ctx)

! Create vector with 10000 double-precision elements
global_size = 10000_c_int64_t
status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
if (status /= DTL_SUCCESS) then
    print *, 'Error creating vector'
    stop 1
end if

! Get local data pointer
local_size = dtl_vector_local_size(vec)
local_offset = dtl_vector_local_offset(vec)
data_ptr = dtl_vector_local_data_mut(vec)

! Convert to Fortran pointer
call c_f_pointer(data_ptr, data, [local_size])

! Fill with values
do i = 1, local_size
    data(i) = real(local_offset + i - 1, kind=c_double)
end do

print '(A,I0,A,I0)', 'Local size: ', local_size, ', offset: ', local_offset

! Cleanup
call dtl_vector_destroy(vec)
call dtl_context_destroy(ctx)
```

### Working with Different Data Types

```fortran
! Integer vector
status = dtl_vector_create(ctx, DTL_DTYPE_INT32, global_size, int_vec)
call c_f_pointer(dtl_vector_local_data_mut(int_vec), int_data, [local_size])

! Single precision
status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT32, global_size, float_vec)
call c_f_pointer(dtl_vector_local_data_mut(float_vec), float_data, [local_size])
```

---

## Collective Operations

### Allreduce

```fortran
use dtl_bindings
implicit none

type(c_ptr) :: ctx
integer(c_int) :: status
real(c_double), target :: local_sum, global_sum

! ... create context and compute local_sum ...

status = dtl_allreduce(ctx, c_loc(local_sum), c_loc(global_sum), &
                       1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_SUM)

if (status == DTL_SUCCESS) then
    print '(A,F12.4)', 'Global sum: ', global_sum
end if
```

### Broadcast

```fortran
real(c_double), target :: data(100)
integer(c_int) :: root

root = 0  ! Root rank

! Root fills data
if (dtl_context_rank(ctx) == root) then
    data = 42.0d0
end if

! Broadcast to all
status = dtl_broadcast(ctx, c_loc(data), 100_c_int64_t, &
                       DTL_DTYPE_FLOAT64, root)
```

### Reduce

```fortran
! Reduce to root
status = dtl_reduce(ctx, c_loc(local_data), c_loc(result), &
                    count, DTL_DTYPE_FLOAT64, DTL_OP_SUM, root)

! Only root has valid result
if (dtl_context_rank(ctx) == root) then
    print *, 'Result: ', result
end if
```

---

## Error Handling

```fortran
integer(c_int) :: status
character(len=:), allocatable :: error_msg

status = dtl_some_operation(...)

if (status /= DTL_SUCCESS) then
    error_msg = get_error_message(status)
    print *, 'DTL Error: ', error_msg

    ! Handle specific errors
    select case (status)
        case (DTL_ERROR_COMMUNICATION)
            print *, 'Communication failure'
        case (DTL_ERROR_MEMORY)
            print *, 'Memory allocation failure'
        case (DTL_ERROR_BOUNDS)
            print *, 'Index out of bounds'
        case default
            print *, 'Unknown error'
    end select

    stop 1
end if
```

---

## Complete Examples

### Example 1: Distributed Sum

```fortran
!> Compute sum of distributed vector
program distributed_sum
    use dtl_bindings
    implicit none

    type(c_ptr) :: ctx, vec
    integer(c_int) :: status, rank
    integer(c_int64_t) :: global_size, local_size, local_offset, i
    real(c_double), pointer :: data(:)
    real(c_double), target :: local_sum, global_sum
    real(c_double) :: expected

    ! Initialize
    status = dtl_context_create_default(ctx)
    if (status /= DTL_SUCCESS) stop 'Context failed'

    rank = dtl_context_rank(ctx)

    ! Create vector
    global_size = 10000_c_int64_t
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, global_size, vec)
    if (status /= DTL_SUCCESS) stop 'Vector failed'

    ! Get local data
    local_size = dtl_vector_local_size(vec)
    local_offset = dtl_vector_local_offset(vec)
    call c_f_pointer(dtl_vector_local_data_mut(vec), data, [local_size])

    ! Fill with indices
    do i = 1, local_size
        data(i) = real(local_offset + i - 1, kind=c_double)
    end do

    ! Compute local sum
    local_sum = sum(data)

    ! Global reduction
    status = dtl_allreduce(ctx, c_loc(local_sum), c_loc(global_sum), &
                           1_c_int64_t, DTL_DTYPE_FLOAT64, DTL_OP_SUM)

    ! Verify (sum of 0..N-1 = N*(N-1)/2)
    expected = real(global_size * (global_size - 1) / 2, kind=c_double)

    if (rank == 0) then
        print '(A,F15.1)', 'Global sum: ', global_sum
        print '(A,F15.1)', 'Expected:   ', expected
        if (abs(global_sum - expected) < 0.01d0) then
            print *, 'SUCCESS!'
        else
            print *, 'FAILURE!'
        end if
    end if

    ! Cleanup
    call dtl_vector_destroy(vec)
    call dtl_context_destroy(ctx)

end program distributed_sum
```

### Example 2: Matrix-Vector Multiply

```fortran
!> Distributed matrix-vector multiply (row-distributed matrix)
program matvec
    use dtl_bindings
    implicit none

    type(c_ptr) :: ctx, x_vec, y_vec
    integer(c_int) :: status, rank, size
    integer(c_int64_t) :: n, local_rows, row_offset, i, j
    real(c_double), pointer :: x(:), y(:)
    real(c_double), allocatable :: local_matrix(:,:), global_x(:)
    real(c_double), target :: temp

    ! Initialize
    status = dtl_context_create_default(ctx)
    rank = dtl_context_rank(ctx)
    size = dtl_context_size(ctx)

    n = 1000_c_int64_t  ! N x N matrix, N vector

    ! Create distributed vectors
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, n, x_vec)
    status = dtl_vector_create(ctx, DTL_DTYPE_FLOAT64, n, y_vec)

    local_rows = dtl_vector_local_size(x_vec)
    row_offset = dtl_vector_local_offset(x_vec)

    ! Get local data pointers
    call c_f_pointer(dtl_vector_local_data_mut(x_vec), x, [local_rows])
    call c_f_pointer(dtl_vector_local_data_mut(y_vec), y, [local_rows])

    ! Initialize x
    do i = 1, local_rows
        x(i) = 1.0d0
    end do

    ! We need full x for matrix-vector multiply
    allocate(global_x(n))

    ! Allgather x (simplified - in practice use dtl_allgather)
    ! Here we just use a simple pattern
    global_x = 1.0d0

    ! Local matrix (row-distributed)
    allocate(local_matrix(local_rows, n))
    do i = 1, local_rows
        do j = 1, n
            ! Simple matrix: A[i,j] = i + j
            local_matrix(i, j) = real(row_offset + i + j, kind=c_double)
        end do
    end do

    ! Compute y = A * x (local rows)
    y = matmul(local_matrix, global_x)

    ! Report
    if (rank == 0) then
        print '(A,I0,A)', 'Computed ', n, ' x ', n, ' matrix-vector product'
        print '(A,F15.4)', 'y(1) = ', y(1)
    end if

    ! Cleanup
    deallocate(local_matrix, global_x)
    call dtl_vector_destroy(x_vec)
    call dtl_vector_destroy(y_vec)
    call dtl_context_destroy(ctx)

end program matvec
```

---

## Building and Running

### CMake Integration (Recommended)

The recommended way to use DTL Fortran bindings is via CMake:

```cmake
find_package(DTL REQUIRED)
add_executable(my_program my_program.f90)
target_link_libraries(my_program PRIVATE DTL::dtl_fortran)
```

Build DTL with Fortran bindings enabled:

```bash
cmake -B build \
    -DDTL_BUILD_C_BINDINGS=ON \
    -DDTL_BUILD_FORTRAN=ON
cmake --build build
cmake --install build
```

### Running

```bash
mpirun -np 4 ./my_program
```

### Legacy Manual Compilation

For environments without CMake:

```bash
# Compile the interface module first
gfortran -c dtl.f90

# Compile your program
gfortran -c my_program.f90

# Link with DTL and MPI
gfortran -o my_program my_program.o dtl.o -ldtl_c -lmpi_mpifh -lmpi
```

Note: The CMake-based build path above is the recommended approach.

---

## References

- [C Bindings Guide](c_bindings.md) - Full C API reference
- [ISO_C_BINDING (Fortran Wiki)](https://fortranwiki.org/fortran/show/iso_c_binding)
