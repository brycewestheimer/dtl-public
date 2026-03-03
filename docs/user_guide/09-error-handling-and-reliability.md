# 9. Error Handling and Reliability

## Error model overview

DTL surfaces errors through status/result patterns and, in some APIs, exception-style or callback-configurable behavior.

## C++ guidance

- prefer explicit `result/status` checking in distributed control paths
- enrich error context near backend boundaries
- avoid suppressing backend or collective participation errors

## C ABI guidance

- check every returned `dtl_status`
- translate status to logs/messages at call boundaries
- treat `*_UNAVAILABLE` and `*_FAILED` distinctly

## Python guidance

- catch binding exceptions and preserve actionable context
- keep async request failures observable at await/wait boundaries

## Reliability practices

1. validate inputs/handles early
2. use explicit synchronization points where required
3. fail fast on collective contract violations
4. ensure deterministic cleanup paths for partially initialized states

## Failure categories to handle explicitly

- invalid argument / null pointer
- backend unavailable or failed initialization
- communication/collective failure
- timeout/cancellation paths (where applicable)

## Operational recommendations

- centralize status-to-message translation in application adapters
- record rank/context identifiers in logs
- include backend capability snapshot in startup diagnostics

## Next step

Proceed to [Chapter 10](10-performance-tuning-and-scaling.md).

## Deep-dive reference

- [Legacy Deep-Dive: Error Handling](error_handling.md)
- [Runtime and Handle Model](13-runtime-and-handle-model.md)
