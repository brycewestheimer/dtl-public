! Copyright (c) 2026 Bryce M. Westheimer
! SPDX-License-Identifier: BSD-3-Clause

!> @file dtl_containers.f90
!> @brief DTL Containers Module - Vector, Array, Tensor, Map, Span
!> @since 0.1.0

module dtl_containers
    use, intrinsic :: iso_c_binding
    use dtl_core, only: dtl_shape
    implicit none

    private

    ! Vector operations
    public :: dtl_vector_create, dtl_vector_create_fill, dtl_vector_destroy
    public :: dtl_vector_global_size, dtl_vector_local_size
    public :: dtl_vector_local_offset
    public :: dtl_vector_local_data, dtl_vector_local_data_mut
    public :: dtl_vector_dtype, dtl_vector_empty
    public :: dtl_vector_fill_local, dtl_vector_barrier, dtl_vector_is_valid
    public :: dtl_vector_get_local, dtl_vector_set_local
    public :: dtl_vector_num_ranks, dtl_vector_rank
    public :: dtl_vector_is_local, dtl_vector_owner
    public :: dtl_vector_to_local, dtl_vector_to_global
    public :: dtl_vector_resize, dtl_vector_redistribute
    public :: dtl_vector_is_dirty, dtl_vector_is_clean, dtl_vector_sync

    ! Array operations
    public :: dtl_array_create, dtl_array_destroy
    public :: dtl_array_global_size, dtl_array_local_size
    public :: dtl_array_local_offset
    public :: dtl_array_local_data, dtl_array_local_data_mut
    public :: dtl_array_dtype, dtl_array_empty
    public :: dtl_array_fill_local, dtl_array_barrier, dtl_array_is_valid

    ! Tensor operations
    public :: dtl_tensor_create, dtl_tensor_create_fill, dtl_tensor_destroy
    public :: dtl_tensor_shape, dtl_tensor_ndim, dtl_tensor_dim
    public :: dtl_tensor_global_size, dtl_tensor_local_size
    public :: dtl_tensor_local_shape, dtl_tensor_dtype
    public :: dtl_tensor_local_data, dtl_tensor_local_data_mut
    public :: dtl_tensor_stride
    public :: dtl_tensor_get_local, dtl_tensor_set_local
    public :: dtl_tensor_get_local_nd, dtl_tensor_set_local_nd
    public :: dtl_tensor_num_ranks, dtl_tensor_rank
    public :: dtl_tensor_distributed_dim
    public :: dtl_tensor_reshape, dtl_tensor_fill_local
    public :: dtl_tensor_barrier, dtl_tensor_is_valid

    ! Span operations
    public :: dtl_span_create, dtl_span_from_vector, dtl_span_from_array
    public :: dtl_span_from_tensor, dtl_span_destroy
    public :: dtl_span_size, dtl_span_local_size, dtl_span_size_bytes
    public :: dtl_span_empty, dtl_span_dtype
    public :: dtl_span_data, dtl_span_data_mut
    public :: dtl_span_rank, dtl_span_num_ranks
    public :: dtl_span_get_local, dtl_span_set_local
    public :: dtl_span_first, dtl_span_last, dtl_span_subspan
    public :: dtl_span_is_valid

    ! ======================================================================
    ! C API Interface Declarations
    ! ======================================================================

    interface

        ! ==================================================================
        ! Vector Operations
        ! ==================================================================

        function dtl_vector_create(ctx, dtype, global_size, vec) &
                bind(c, name='dtl_vector_create')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: global_size
            type(c_ptr), intent(out) :: vec
            integer(c_int) :: dtl_vector_create
        end function

        function dtl_vector_create_fill(ctx, dtype, global_size, value, vec) &
                bind(c, name='dtl_vector_create_fill')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: global_size
            type(c_ptr), value :: value
            type(c_ptr), intent(out) :: vec
            integer(c_int) :: dtl_vector_create_fill
        end function

        subroutine dtl_vector_destroy(vec) bind(c, name='dtl_vector_destroy')
            import :: c_ptr
            type(c_ptr), value :: vec
        end subroutine

        function dtl_vector_global_size(vec) &
                bind(c, name='dtl_vector_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_global_size
        end function

        function dtl_vector_local_size(vec) &
                bind(c, name='dtl_vector_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_size
        end function

        function dtl_vector_local_offset(vec) &
                bind(c, name='dtl_vector_local_offset')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t) :: dtl_vector_local_offset
        end function

        function dtl_vector_local_data(vec) &
                bind(c, name='dtl_vector_local_data')
            import :: c_ptr
            type(c_ptr), value :: vec
            type(c_ptr) :: dtl_vector_local_data
        end function

        function dtl_vector_local_data_mut(vec) &
                bind(c, name='dtl_vector_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: vec
            type(c_ptr) :: dtl_vector_local_data_mut
        end function

        function dtl_vector_dtype(vec) bind(c, name='dtl_vector_dtype')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_dtype
        end function

        function dtl_vector_empty(vec) bind(c, name='dtl_vector_empty')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_empty
        end function

        function dtl_vector_fill_local(vec, value) &
                bind(c, name='dtl_vector_fill_local')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            type(c_ptr), value :: value
            integer(c_int) :: dtl_vector_fill_local
        end function

        function dtl_vector_barrier(vec) bind(c, name='dtl_vector_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_barrier
        end function

        function dtl_vector_is_valid(vec) bind(c, name='dtl_vector_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_is_valid
        end function

        function dtl_vector_get_local(vec, local_idx, value) &
                bind(c, name='dtl_vector_get_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: local_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_vector_get_local
        end function

        function dtl_vector_set_local(vec, local_idx, value) &
                bind(c, name='dtl_vector_set_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: local_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_vector_set_local
        end function

        function dtl_vector_num_ranks(vec) &
                bind(c, name='dtl_vector_num_ranks')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_num_ranks
        end function

        function dtl_vector_rank(vec) bind(c, name='dtl_vector_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_rank
        end function

        function dtl_vector_is_local(vec, global_idx) &
                bind(c, name='dtl_vector_is_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: global_idx
            integer(c_int) :: dtl_vector_is_local
        end function

        function dtl_vector_owner(vec, global_idx) &
                bind(c, name='dtl_vector_owner')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: global_idx
            integer(c_int) :: dtl_vector_owner
        end function

        function dtl_vector_to_local(vec, global_idx) &
                bind(c, name='dtl_vector_to_local')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: global_idx
            integer(c_int64_t) :: dtl_vector_to_local
        end function

        function dtl_vector_to_global(vec, local_idx) &
                bind(c, name='dtl_vector_to_global')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: local_idx
            integer(c_int64_t) :: dtl_vector_to_global
        end function

        function dtl_vector_resize(vec, new_size) &
                bind(c, name='dtl_vector_resize')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: vec
            integer(c_int64_t), value :: new_size
            integer(c_int) :: dtl_vector_resize
        end function

        function dtl_vector_redistribute(vec, new_partition) &
                bind(c, name='dtl_vector_redistribute')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int), value :: new_partition
            integer(c_int) :: dtl_vector_redistribute
        end function

        function dtl_vector_is_dirty(vec) &
                bind(c, name='dtl_vector_is_dirty')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_is_dirty
        end function

        function dtl_vector_is_clean(vec) &
                bind(c, name='dtl_vector_is_clean')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_is_clean
        end function

        function dtl_vector_sync(vec) bind(c, name='dtl_vector_sync')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            integer(c_int) :: dtl_vector_sync
        end function

        ! ==================================================================
        ! Array Operations
        ! ==================================================================

        function dtl_array_create(ctx, dtype, size, arr) &
                bind(c, name='dtl_array_create')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            integer(c_int64_t), value :: size
            type(c_ptr), intent(out) :: arr
            integer(c_int) :: dtl_array_create
        end function

        subroutine dtl_array_destroy(arr) bind(c, name='dtl_array_destroy')
            import :: c_ptr
            type(c_ptr), value :: arr
        end subroutine

        function dtl_array_global_size(arr) &
                bind(c, name='dtl_array_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            integer(c_int64_t) :: dtl_array_global_size
        end function

        function dtl_array_local_size(arr) &
                bind(c, name='dtl_array_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            integer(c_int64_t) :: dtl_array_local_size
        end function

        function dtl_array_local_offset(arr) &
                bind(c, name='dtl_array_local_offset')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: arr
            integer(c_int64_t) :: dtl_array_local_offset
        end function

        function dtl_array_local_data(arr) &
                bind(c, name='dtl_array_local_data')
            import :: c_ptr
            type(c_ptr), value :: arr
            type(c_ptr) :: dtl_array_local_data
        end function

        function dtl_array_local_data_mut(arr) &
                bind(c, name='dtl_array_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: arr
            type(c_ptr) :: dtl_array_local_data_mut
        end function

        function dtl_array_dtype(arr) bind(c, name='dtl_array_dtype')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_dtype
        end function

        function dtl_array_empty(arr) bind(c, name='dtl_array_empty')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_empty
        end function

        function dtl_array_fill_local(arr, value) &
                bind(c, name='dtl_array_fill_local')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            type(c_ptr), value :: value
            integer(c_int) :: dtl_array_fill_local
        end function

        function dtl_array_barrier(arr) bind(c, name='dtl_array_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_barrier
        end function

        function dtl_array_is_valid(arr) bind(c, name='dtl_array_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            integer(c_int) :: dtl_array_is_valid
        end function

        ! ==================================================================
        ! Tensor Operations
        ! ==================================================================

        function dtl_tensor_create(ctx, dtype, shape, tensor) &
                bind(c, name='dtl_tensor_create')
            import :: c_ptr, c_int, dtl_shape
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            type(dtl_shape), value :: shape
            type(c_ptr), intent(out) :: tensor
            integer(c_int) :: dtl_tensor_create
        end function

        function dtl_tensor_create_fill(ctx, dtype, shape, value, tensor) &
                bind(c, name='dtl_tensor_create_fill')
            import :: c_ptr, c_int, dtl_shape
            type(c_ptr), value :: ctx
            integer(c_int), value :: dtype
            type(dtl_shape), value :: shape
            type(c_ptr), value :: value
            type(c_ptr), intent(out) :: tensor
            integer(c_int) :: dtl_tensor_create_fill
        end function

        subroutine dtl_tensor_destroy(tensor) &
                bind(c, name='dtl_tensor_destroy')
            import :: c_ptr
            type(c_ptr), value :: tensor
        end subroutine

        function dtl_tensor_shape(tensor) bind(c, name='dtl_tensor_shape')
            import :: c_ptr, dtl_shape
            type(c_ptr), value :: tensor
            type(dtl_shape) :: dtl_tensor_shape
        end function

        function dtl_tensor_ndim(tensor) bind(c, name='dtl_tensor_ndim')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_ndim
        end function

        function dtl_tensor_dim(tensor, dim) bind(c, name='dtl_tensor_dim')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int), value :: dim
            integer(c_int64_t) :: dtl_tensor_dim
        end function

        function dtl_tensor_global_size(tensor) &
                bind(c, name='dtl_tensor_global_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t) :: dtl_tensor_global_size
        end function

        function dtl_tensor_local_size(tensor) &
                bind(c, name='dtl_tensor_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t) :: dtl_tensor_local_size
        end function

        function dtl_tensor_local_shape(tensor) &
                bind(c, name='dtl_tensor_local_shape')
            import :: c_ptr, dtl_shape
            type(c_ptr), value :: tensor
            type(dtl_shape) :: dtl_tensor_local_shape
        end function

        function dtl_tensor_dtype(tensor) bind(c, name='dtl_tensor_dtype')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_dtype
        end function

        function dtl_tensor_local_data(tensor) &
                bind(c, name='dtl_tensor_local_data')
            import :: c_ptr
            type(c_ptr), value :: tensor
            type(c_ptr) :: dtl_tensor_local_data
        end function

        function dtl_tensor_local_data_mut(tensor) &
                bind(c, name='dtl_tensor_local_data_mut')
            import :: c_ptr
            type(c_ptr), value :: tensor
            type(c_ptr) :: dtl_tensor_local_data_mut
        end function

        function dtl_tensor_stride(tensor, dim) &
                bind(c, name='dtl_tensor_stride')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int), value :: dim
            integer(c_int64_t) :: dtl_tensor_stride
        end function

        function dtl_tensor_get_local(tensor, linear_idx, value) &
                bind(c, name='dtl_tensor_get_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t), value :: linear_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_tensor_get_local
        end function

        function dtl_tensor_set_local(tensor, linear_idx, value) &
                bind(c, name='dtl_tensor_set_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t), value :: linear_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_tensor_set_local
        end function

        function dtl_tensor_get_local_nd(tensor, indices, value) &
                bind(c, name='dtl_tensor_get_local_nd')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t), intent(in) :: indices(*)
            type(c_ptr), value :: value
            integer(c_int) :: dtl_tensor_get_local_nd
        end function

        function dtl_tensor_set_local_nd(tensor, indices, value) &
                bind(c, name='dtl_tensor_set_local_nd')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: tensor
            integer(c_int64_t), intent(in) :: indices(*)
            type(c_ptr), value :: value
            integer(c_int) :: dtl_tensor_set_local_nd
        end function

        function dtl_tensor_num_ranks(tensor) &
                bind(c, name='dtl_tensor_num_ranks')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_num_ranks
        end function

        function dtl_tensor_rank(tensor) bind(c, name='dtl_tensor_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_rank
        end function

        function dtl_tensor_distributed_dim(tensor) &
                bind(c, name='dtl_tensor_distributed_dim')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_distributed_dim
        end function

        function dtl_tensor_reshape(tensor, new_shape) &
                bind(c, name='dtl_tensor_reshape')
            import :: c_ptr, c_int, dtl_shape
            type(c_ptr), value :: tensor
            type(dtl_shape), value :: new_shape
            integer(c_int) :: dtl_tensor_reshape
        end function

        function dtl_tensor_fill_local(tensor, value) &
                bind(c, name='dtl_tensor_fill_local')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            type(c_ptr), value :: value
            integer(c_int) :: dtl_tensor_fill_local
        end function

        function dtl_tensor_barrier(tensor) &
                bind(c, name='dtl_tensor_barrier')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_barrier
        end function

        function dtl_tensor_is_valid(tensor) &
                bind(c, name='dtl_tensor_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            integer(c_int) :: dtl_tensor_is_valid
        end function

        ! ==================================================================
        ! Span Operations
        ! ==================================================================

        function dtl_span_create(dtype, local_data, local_size, global_size, &
                                 rank, num_ranks, span) &
                bind(c, name='dtl_span_create')
            import :: c_ptr, c_int, c_int64_t
            integer(c_int), value :: dtype
            type(c_ptr), value :: local_data
            integer(c_int64_t), value :: local_size
            integer(c_int64_t), value :: global_size
            integer(c_int), value :: rank
            integer(c_int), value :: num_ranks
            type(c_ptr), intent(out) :: span
            integer(c_int) :: dtl_span_create
        end function

        function dtl_span_from_vector(vec, span) &
                bind(c, name='dtl_span_from_vector')
            import :: c_ptr, c_int
            type(c_ptr), value :: vec
            type(c_ptr), intent(out) :: span
            integer(c_int) :: dtl_span_from_vector
        end function

        function dtl_span_from_array(arr, span) &
                bind(c, name='dtl_span_from_array')
            import :: c_ptr, c_int
            type(c_ptr), value :: arr
            type(c_ptr), intent(out) :: span
            integer(c_int) :: dtl_span_from_array
        end function

        function dtl_span_from_tensor(tensor, span) &
                bind(c, name='dtl_span_from_tensor')
            import :: c_ptr, c_int
            type(c_ptr), value :: tensor
            type(c_ptr), intent(out) :: span
            integer(c_int) :: dtl_span_from_tensor
        end function

        subroutine dtl_span_destroy(span) bind(c, name='dtl_span_destroy')
            import :: c_ptr
            type(c_ptr), value :: span
        end subroutine

        function dtl_span_size(span) bind(c, name='dtl_span_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t) :: dtl_span_size
        end function

        function dtl_span_local_size(span) &
                bind(c, name='dtl_span_local_size')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t) :: dtl_span_local_size
        end function

        function dtl_span_size_bytes(span) &
                bind(c, name='dtl_span_size_bytes')
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t) :: dtl_span_size_bytes
        end function

        function dtl_span_empty(span) bind(c, name='dtl_span_empty')
            import :: c_ptr, c_int
            type(c_ptr), value :: span
            integer(c_int) :: dtl_span_empty
        end function

        function dtl_span_dtype(span) bind(c, name='dtl_span_dtype')
            import :: c_ptr, c_int
            type(c_ptr), value :: span
            integer(c_int) :: dtl_span_dtype
        end function

        function dtl_span_data(span) bind(c, name='dtl_span_data')
            import :: c_ptr
            type(c_ptr), value :: span
            type(c_ptr) :: dtl_span_data
        end function

        function dtl_span_data_mut(span) bind(c, name='dtl_span_data_mut')
            import :: c_ptr
            type(c_ptr), value :: span
            type(c_ptr) :: dtl_span_data_mut
        end function

        function dtl_span_rank(span) bind(c, name='dtl_span_rank')
            import :: c_ptr, c_int
            type(c_ptr), value :: span
            integer(c_int) :: dtl_span_rank
        end function

        function dtl_span_num_ranks(span) bind(c, name='dtl_span_num_ranks')
            import :: c_ptr, c_int
            type(c_ptr), value :: span
            integer(c_int) :: dtl_span_num_ranks
        end function

        function dtl_span_get_local(span, local_idx, value) &
                bind(c, name='dtl_span_get_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t), value :: local_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_span_get_local
        end function

        function dtl_span_set_local(span, local_idx, value) &
                bind(c, name='dtl_span_set_local')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t), value :: local_idx
            type(c_ptr), value :: value
            integer(c_int) :: dtl_span_set_local
        end function

        function dtl_span_first(span, count, out_span) &
                bind(c, name='dtl_span_first')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t), value :: count
            type(c_ptr), intent(out) :: out_span
            integer(c_int) :: dtl_span_first
        end function

        function dtl_span_last(span, count, out_span) &
                bind(c, name='dtl_span_last')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t), value :: count
            type(c_ptr), intent(out) :: out_span
            integer(c_int) :: dtl_span_last
        end function

        function dtl_span_subspan(span, offset, count, out_span) &
                bind(c, name='dtl_span_subspan')
            import :: c_ptr, c_int, c_int64_t
            type(c_ptr), value :: span
            integer(c_int64_t), value :: offset
            integer(c_int64_t), value :: count
            type(c_ptr), intent(out) :: out_span
            integer(c_int) :: dtl_span_subspan
        end function

        function dtl_span_is_valid(span) bind(c, name='dtl_span_is_valid')
            import :: c_ptr, c_int
            type(c_ptr), value :: span
            integer(c_int) :: dtl_span_is_valid
        end function

    end interface

end module dtl_containers
