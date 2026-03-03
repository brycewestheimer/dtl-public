# Spack Repository

DTL ships a local Spack repository in this directory.

## Add the repository

```bash
spack repo add ./spack
```

Verify that Spack can see the package:

```bash
spack repo list
spack info dtl
```

## Common installs

```bash
spack install dtl
spack install dtl +tests
spack install dtl +python +c_bindings
spack install dtl +fortran +c_bindings
spack install dtl +docs
spack install dtl +cuda +nccl
```

## Notes

- `+python` requires `+c_bindings` because the Python module links the C ABI.
- `+fortran` requires `+c_bindings` for the same reason.
- `+docs` builds the Doxygen + Sphinx documentation site.
- `+integration_tests` requires `+tests`.

## Editable development builds

```bash
spack repo add ./spack
spack dev-build dtl@main
```
