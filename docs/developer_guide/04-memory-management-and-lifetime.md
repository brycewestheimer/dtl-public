# 4. Memory Management and Lifetime

## Ownership model

Use explicit ownership categories:

- `std::unique_ptr` for exclusive ownership
- `std::shared_ptr` only when shared lifetime is required
- raw pointer/reference for non-owning access only

## Rules

- prefer RAII for all resource lifetimes
- avoid raw `new`/`delete` in ordinary C++ ownership flows
- if raw handles are required at ABI boundaries, wrap internals in RAII
- document lifetime assumptions for borrowed pointers

## Non-owning span/view rules

`dtl::distributed_span` and binding-layer local views are borrowed views:

- never transfer ownership through span/view construction
- ensure the owning container/handle outlives every borrowed span/view
- refresh span/view objects after structural operations (`resize`, `redistribute`, or owner recreation)

## ABI boundary exceptions

C ABI handles are opaque pointer types by design. This is acceptable when:

- creation and destroy APIs are paired and tested
- validation/magic checks exist at boundaries
- cleanup is deterministic on all error paths

## Python binding lifetime rules

- keep native handles in wrapper classes with deterministic destruction
- preserve Python object ownership for NumPy view lifetimes
- avoid leaking callback/request state on async paths

## Fortran binding lifetime rules

- keep ownership in C ABI layer
- Fortran holds `type(c_ptr)` handles and must call destroy functions
- ensure wrappers do not copy or orphan owning pointers

## Memory review checklist

- no ownership cycles without reason
- no leaked request/window/context handles
- no dangling view owners
- all early-return error paths free allocated resources
