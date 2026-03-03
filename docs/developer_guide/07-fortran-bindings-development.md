# 7. Fortran Bindings Development

## Structure

- Module source: `bindings/fortran/dtl.f90`
- Tests: `bindings/fortran/tests/`
- Examples: `bindings/fortran/examples/`

## Design principles

- Fortran bindings mirror C ABI behavior
- interop uses `iso_c_binding`
- ownership remains in C ABI layer

## Interface rules

- prefer explicit `bind(c, name='...')` interfaces
- use correct `c_ptr`, `c_int`, `c_size_t` compatible types
- keep status handling aligned with C ABI contract

## Lifetime expectations

- Fortran code must call destroy functions for created handles
- wrappers should not assume automatic cleanup of C resources
- `dtl_span_*` handles are non-owning and must not outlive owner containers
- `c_ptr` + `c_f_pointer` local projections from span/container handles are borrowed views and must not outlive the owning C handle

## Validation

Build and run Fortran smoke tests:

```bash
cmake --build <build-dir> -j6 --target test_fortran_basic
ctest --test-dir <build-dir> -R '^fortran_basic_test$' -j6 --output-on-failure
```

## When changing C ABI

If signatures/status behavior change in C:

1. update `dtl.f90` interfaces
2. update Fortran tests/examples
3. rerun Fortran smoke tests

## Related chapter

- Runtime/handle development: `docs/developer_guide/12-runtime-and-handle-development.md`
