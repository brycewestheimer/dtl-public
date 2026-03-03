# Fortran API Reference

**DTL Version:** 0.1.0-alpha.1
**Last Updated:** 2026-03-03

The supported Fortran surface is the native `dtl` module built by
`DTL_BUILD_FORTRAN=ON`. The canonical examples for this alpha release live in
`bindings/fortran/examples`.

## Build Preconditions

- `DTL_BUILD_FORTRAN=ON` requires `DTL_BUILD_C_BINDINGS=ON`
- A Fortran compiler must be available
- MPI-dependent behavior requires a build with MPI support and an MPI runtime

## Ownership and Lifetime

- The `dtl` module wraps the C API through `ISO_C_BINDING`
- Handles returned as `type(c_ptr)` are owning unless documented otherwise
- Span handles are non-owning; the source container must outlive derived spans
- `c_f_pointer` projections remain valid only while the underlying DTL object remains alive and unmoved

## Core Entry Points

Version and feature queries:

```fortran
dtl_version_major()
dtl_version_minor()
dtl_version_patch()
dtl_has_mpi()
dtl_has_cuda()
```

Context management:

```fortran
dtl_context_create_default(ctx)
dtl_context_destroy(ctx)
dtl_context_rank(ctx)
dtl_context_size(ctx)
dtl_context_is_root(ctx)
dtl_barrier(ctx)
```

Containers:

```fortran
dtl_vector_create(ctx, dtype, size, vec)
dtl_vector_destroy(vec)
dtl_vector_local_data_mut(vec)

dtl_tensor_create(ctx, dtype, shape, tensor)
dtl_tensor_destroy(tensor)

dtl_span_from_vector(vec, span)
dtl_span_destroy(span)
```

Collectives:

```fortran
dtl_broadcast(ctx, data, count, dtype, root)
dtl_reduce(ctx, sendbuf, recvbuf, count, dtype, op, root)
dtl_allreduce(ctx, sendbuf, recvbuf, count, dtype, op)
```

## Example Build Path

```bash
cmake -S . -B build \
  -DDTL_BUILD_C_BINDINGS=ON \
  -DDTL_BUILD_FORTRAN=ON \
  -DDTL_BUILD_EXAMPLES=ON
cmake --build build
```

Expected example binaries for this alpha release:

- `build/bin/fortran/dtl_fortran_hello`
- `build/bin/fortran/dtl_fortran_vector_demo`
