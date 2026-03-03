# 6. Python Bindings Development

## Structure

- Native extension: `bindings/python/src/dtl/*.cpp`
- High-level API: `bindings/python/src/dtl/__init__.py`
- Tests: `bindings/python/tests/`

## Layering model

1. pybind11 C++ layer exposes low-level native operations
2. `__init__.py` provides Pythonic wrappers and compatibility helpers
3. tests validate behavior, versions, and edge cases

## API design expectations

- preserve stable high-level API names and signatures when possible
- keep version metadata consistent with package/native version sources
- avoid hidden behavior differences between sync and async APIs

## Lifetime and memory safety

- use RAII wrappers for native request/window/handle ownership
- ensure callbacks do not outlive owning Python objects unsafely
- when exposing NumPy views, keep the owner object alive

## `distributed_span` in Python

Python exposes a first-class `DistributedSpan` factory and typed native wrappers (`DistributedSpan_*`).

- source owners must be retained in span wrappers to prevent dangling storage
- NumPy `local_view()` for spans must keep span wrapper ownership alive
- avoid caching span-derived local views across structural owner changes

## Running tests

Use non-MPI/non-CUDA sweep when validating broad API behavior:

```bash
MPI4PY_RC_INITIALIZE=0 \
PYTHONPATH=<build-dir>/bindings/python \
python3 -m pytest bindings/python/tests -q -m 'not mpi and not cuda'
```

## Common pitfalls

- importing wrong module path during local test runs
- mismatched type conversions for scalar dtypes
- async handles returned without clear lifecycle wrappers

## Related chapter

- Runtime/handle development: `docs/developer_guide/12-runtime-and-handle-development.md`
