# 5. C ABI Development

## Scope

C ABI headers are under `include/dtl/bindings/c/` and implementation is under `src/bindings/c/`.

## ABI design rules

- opaque handle types only
- explicit `dtl_status` returns for fallible APIs
- no C++ exceptions crossing C boundaries
- null/invalid handle checks at entry points
- treat local buffer accessors and `dtl_span_t` as non-owning borrowed views, never as owning transfers

## Backend availability behavior

When requested functionality depends on unavailable backends, return explicit availability errors such as `DTL_ERROR_BACKEND_UNAVAILABLE`.

## Handle lifecycle

- create/init functions assign valid handles on success
- destroy/free functions are idempotent where possible
- every allocation path has cleanup on failure
- runtime capability gating and handle validity must remain consistent

## API evolution

When adding or changing C ABI calls:

1. update public C headers with complete Doxygen comments
2. implement in `src/bindings/c/`
3. add/extend C ABI tests
4. update language binding adapters if behavior changed

## Testing

Build and run C ABI checks:

```bash
cmake --build <build-dir> -j6 --target test_c_bindings
ctest --test-dir <build-dir> -R '^CBindingsTests$' -j6 --output-on-failure
```

## Related chapter

- Runtime/handle development: `docs/developer_guide/12-runtime-and-handle-development.md`
