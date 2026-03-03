# 3. C++ Contribution Standards

## Baseline coding expectations

- C++20 style and concepts where applicable
- clear ownership semantics
- no hidden communication side effects in local APIs
- explicit distributed behavior in API contracts

## Public API discipline

For public headers:

- avoid unnecessary breaking signature changes
- use descriptive parameter names
- maintain consistency with specs and ADRs

## Error handling model

- use `dtl::result<T>`/`dtl::status` where expected by module conventions
- do not silently swallow backend failures
- preserve clear distinction between unsupported and failed operations

## Performance and correctness

- avoid introducing extra copies in hot paths
- maintain local-vs-remote behavioral clarity
- keep collective behavior explicit and contract-based

## Review checklist for C++ changes

- signatures and docs synchronized
- behavior verified with tests
- no ownership regressions
- release notes/spec updates when externally visible
