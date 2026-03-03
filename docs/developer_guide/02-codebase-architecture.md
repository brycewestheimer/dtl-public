# 2. Codebase Architecture

## High-level layout

- `include/dtl/`: primary public C++ API and implementation headers
- `src/`: compiled components, including C ABI implementation units
- `bindings/python/`: pybind11 extension and high-level Python wrapper
- `bindings/fortran/`: ISO C binding module and tests
- `runtime/`: runtime registry and shared runtime support library
- `tests/`: C++ and integration tests
- `docs/`: user docs, specs, ADRs, API references

## Layers

1. Core types, status/result, policy model
2. Containers/views/algorithms
3. Communication and remote/RMA paths
4. Bindings and language adapters
5. Tooling/docs/release surfaces

## Change impact map

When changing a core contract, check all of:

- C++ public headers in `include/dtl/`
- C ABI mirrors in `include/dtl/bindings/c/` and `src/bindings/c/`
- Python wrappers in `bindings/python/src/dtl/`
- Fortran signatures in `bindings/fortran/dtl.f90`
- runtime and handle lifecycle behavior in `runtime/` and binding wrappers
- release notes

## Compatibility expectations

- Keep C++ API behavior stable unless explicitly versioned
- Keep C ABI status-code behavior deterministic
- Preserve Python and Fortran semantic parity where feasible
- Document any intentional incompatibilities before merge

## Runtime/handle references

- Runtime and handle development standards: `docs/developer_guide/12-runtime-and-handle-development.md`
