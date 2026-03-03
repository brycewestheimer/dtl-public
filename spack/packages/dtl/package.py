# Copyright (c) 2026 Bryce M. Westheimer
# SPDX-License-Identifier: BSD-3-Clause
#
# Spack package recipe for DTL (Distributed Template Library)
#
# Usage:
#   spack repo add ./spack
#   spack install dtl
#   spack install dtl +cuda +nccl
#   spack install dtl +python +c_bindings
#   spack install dtl +docs
#
# For local development with an editable checkout:
#   spack repo add ./spack
#   spack dev-build dtl@main

from spack.package import *


class Dtl(CMakePackage):
    """DTL (Distributed Template Library) is a C++20 header-first template
    library providing STL-inspired abstractions for distributed and
    heterogeneous computing."""

    homepage = "https://github.com/brycewestheimer/dtl-public"
    git = "https://github.com/brycewestheimer/dtl-public.git"
    maintainers("brycewestheimer")

    license("BSD-3-Clause")

    version("main", branch="main")
    version("0.1.0-alpha.1", commit="e1c03ccc994e50388f6981fd4d36ff9aefca36d2")

    variant("mpi", default=True, description="Enable MPI backend")
    variant("cuda", default=False, description="Enable CUDA backend")
    variant("hip", default=False, description="Enable HIP backend")
    variant("nccl", default=False, description="Enable NCCL communicator backend")
    variant("shmem", default=False, description="Enable OpenSHMEM backend")

    variant("python", default=False, description="Build Python bindings")
    variant("c_bindings", default=False, description="Build the C ABI library")
    variant("fortran", default=False, description="Build Fortran bindings")

    variant("cereal", default=False, description="Enable Cereal serialization adapter")
    variant(
        "boost_serialization",
        default=False,
        description="Enable Boost.Serialization adapter",
    )

    variant("tests", default=False, description="Build unit tests")
    variant("integration_tests", default=False, description="Build integration tests")
    variant("benchmarks", default=False, description="Build performance benchmarks")
    variant("examples", default=True, description="Build example programs")
    variant("docs", default=False, description="Build documentation site")

    depends_on("cmake@3.20:", type="build")

    depends_on("mpi", when="+mpi")

    depends_on("cuda@11.4:", when="+cuda")
    depends_on("hip", when="+hip")
    depends_on("nccl", when="+nccl")
    depends_on("openshmem", when="+shmem")

    depends_on("python@3.8:", when="+python", type=("build", "run"))
    depends_on("py-pybind11", when="+python", type="build")
    depends_on("py-numpy", when="+python", type=("build", "run"))
    depends_on("py-mpi4py", when="+python+mpi", type=("build", "run"))

    depends_on("cereal", when="+cereal")
    depends_on("boost+serialization", when="+boost_serialization")

    depends_on("googletest@1.12:", when="+tests")
    depends_on("benchmark@1.7:", when="+benchmarks")

    depends_on("python@3.8:", when="+docs", type=("build", "run"))
    depends_on("doxygen", when="+docs", type="build")
    depends_on("graphviz", when="+docs", type="build")
    depends_on("py-sphinx@7:", when="+docs", type="build")
    depends_on("py-sphinx-rtd-theme@2:", when="+docs", type="build")
    depends_on("py-myst-parser@2:", when="+docs", type="build")
    depends_on("py-breathe@4.35:", when="+docs", type="build")

    conflicts("+nccl", when="~cuda ~hip", msg="NCCL requires +cuda or +hip")
    conflicts("+cuda", when="+hip", msg="CUDA and HIP are mutually exclusive")
    conflicts("+integration_tests", when="~tests", msg="Integration tests require +tests")
    conflicts("+python", when="~c_bindings", msg="Python bindings require +c_bindings")
    conflicts("+fortran", when="~c_bindings", msg="Fortran bindings require +c_bindings")

    conflicts("%gcc@:10", msg="DTL requires GCC 11+ for C++20 support")
    conflicts("%clang@:14", msg="DTL requires Clang 15+ for C++20 support")

    build_time_test_callbacks = ["check"]

    def cmake_args(self):
        args = [
            self.define_from_variant("DTL_ENABLE_MPI", "mpi"),
            self.define_from_variant("DTL_ENABLE_CUDA", "cuda"),
            self.define_from_variant("DTL_ENABLE_HIP", "hip"),
            self.define_from_variant("DTL_ENABLE_NCCL", "nccl"),
            self.define_from_variant("DTL_ENABLE_SHMEM", "shmem"),
            self.define_from_variant("DTL_BUILD_PYTHON", "python"),
            self.define_from_variant("DTL_BUILD_C_BINDINGS", "c_bindings"),
            self.define_from_variant("DTL_BUILD_FORTRAN", "fortran"),
            self.define_from_variant("DTL_ENABLE_CEREAL", "cereal"),
            self.define_from_variant(
                "DTL_ENABLE_BOOST_SERIALIZATION", "boost_serialization"
            ),
            self.define_from_variant("DTL_BUILD_TESTS", "tests"),
            self.define_from_variant("DTL_BUILD_INTEGRATION_TESTS", "integration_tests"),
            self.define_from_variant("DTL_BUILD_BENCHMARKS", "benchmarks"),
            self.define_from_variant("DTL_BUILD_EXAMPLES", "examples"),
            self.define_from_variant("DTL_BUILD_DOCS", "docs"),
        ]

        if "+tests" in self.spec:
            args.append(self.define("DTL_USE_SYSTEM_GTEST", True))

        if "+cuda" in self.spec:
            cuda_arch = self.spec.variants.get("cuda_arch", None)
            if cuda_arch and cuda_arch.value:
                args.append(
                    self.define(
                        "CMAKE_CUDA_ARCHITECTURES",
                        ";".join(str(arch) for arch in cuda_arch.value),
                    )
                )

        return args

    def check(self):
        if "+tests" not in self.spec:
            return

        with working_dir(self.build_directory):
            ctest = which("ctest")
            ctest("--output-on-failure", "-L", "smoke")
