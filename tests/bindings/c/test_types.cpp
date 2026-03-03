// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file test_types.cpp
 * @brief Unit tests for DTL C bindings type helpers
 * @since 0.1.0
 */

#include <gtest/gtest.h>
#include <dtl/bindings/c/dtl_types.h>
#include <dtl/bindings/c/dtl_config.h>
#include <cstring>

// ============================================================================
// Version Query Tests
// ============================================================================

TEST(CBindingsTypes, VersionMajorMatches) {
    EXPECT_EQ(dtl_version_major(), DTL_VERSION_MAJOR);
}

TEST(CBindingsTypes, VersionMinorMatches) {
    EXPECT_EQ(dtl_version_minor(), DTL_VERSION_MINOR);
}

TEST(CBindingsTypes, VersionPatchMatches) {
    EXPECT_EQ(dtl_version_patch(), DTL_VERSION_PATCH);
}

TEST(CBindingsTypes, ABIVersionMatches) {
    EXPECT_EQ(dtl_abi_version(), DTL_ABI_VERSION);
}

TEST(CBindingsTypes, VersionStringNotNull) {
    const char* version = dtl_version_string();
    EXPECT_NE(version, nullptr);
    EXPECT_GT(strlen(version), 0u);
}

TEST(CBindingsTypes, VersionStringMatches) {
    EXPECT_STREQ(dtl_version_string(), DTL_VERSION_STRING);
}

// ============================================================================
// Feature Query Tests
// ============================================================================

TEST(CBindingsTypes, HasMpiReturnsValidBool) {
    int has_mpi = dtl_has_mpi();
    EXPECT_TRUE(has_mpi == 0 || has_mpi == 1);
}

TEST(CBindingsTypes, HasCudaReturnsValidBool) {
    int has_cuda = dtl_has_cuda();
    EXPECT_TRUE(has_cuda == 0 || has_cuda == 1);
}

TEST(CBindingsTypes, HasHipReturnsValidBool) {
    int has_hip = dtl_has_hip();
    EXPECT_TRUE(has_hip == 0 || has_hip == 1);
}

TEST(CBindingsTypes, HasNcclReturnsValidBool) {
    int has_nccl = dtl_has_nccl();
    EXPECT_TRUE(has_nccl == 0 || has_nccl == 1);
}

TEST(CBindingsTypes, HasShmemReturnsValidBool) {
    int has_shmem = dtl_has_shmem();
    EXPECT_TRUE(has_shmem == 0 || has_shmem == 1);
}

// ============================================================================
// Data Type Size Tests
// ============================================================================

TEST(CBindingsTypes, DtypeSizeInt8) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_INT8), sizeof(int8_t));
}

TEST(CBindingsTypes, DtypeSizeInt16) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_INT16), sizeof(int16_t));
}

TEST(CBindingsTypes, DtypeSizeInt32) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_INT32), sizeof(int32_t));
}

TEST(CBindingsTypes, DtypeSizeInt64) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_INT64), sizeof(int64_t));
}

TEST(CBindingsTypes, DtypeSizeUint8) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_UINT8), sizeof(uint8_t));
}

TEST(CBindingsTypes, DtypeSizeUint16) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_UINT16), sizeof(uint16_t));
}

TEST(CBindingsTypes, DtypeSizeUint32) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_UINT32), sizeof(uint32_t));
}

TEST(CBindingsTypes, DtypeSizeUint64) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_UINT64), sizeof(uint64_t));
}

TEST(CBindingsTypes, DtypeSizeFloat32) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_FLOAT32), sizeof(float));
}

TEST(CBindingsTypes, DtypeSizeFloat64) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_FLOAT64), sizeof(double));
}

TEST(CBindingsTypes, DtypeSizeByte) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_BYTE), 1u);
}

TEST(CBindingsTypes, DtypeSizeBool) {
    EXPECT_EQ(dtl_dtype_size(DTL_DTYPE_BOOL), sizeof(uint8_t));
}

TEST(CBindingsTypes, DtypeSizeInvalidReturnsZero) {
    EXPECT_EQ(dtl_dtype_size((dtl_dtype)999), 0u);
}

// ============================================================================
// Data Type Name Tests
// ============================================================================

TEST(CBindingsTypes, DtypeNameInt32) {
    EXPECT_STREQ(dtl_dtype_name(DTL_DTYPE_INT32), "int32");
}

TEST(CBindingsTypes, DtypeNameFloat64) {
    EXPECT_STREQ(dtl_dtype_name(DTL_DTYPE_FLOAT64), "float64");
}

TEST(CBindingsTypes, DtypeNameByte) {
    EXPECT_STREQ(dtl_dtype_name(DTL_DTYPE_BYTE), "byte");
}

TEST(CBindingsTypes, DtypeNameInvalidReturnsUnknown) {
    EXPECT_STREQ(dtl_dtype_name((dtl_dtype)999), "unknown");
}

// ============================================================================
// Reduce Op Name Tests
// ============================================================================

TEST(CBindingsTypes, ReduceOpNameSum) {
    EXPECT_STREQ(dtl_reduce_op_name(DTL_OP_SUM), "sum");
}

TEST(CBindingsTypes, ReduceOpNameProd) {
    EXPECT_STREQ(dtl_reduce_op_name(DTL_OP_PROD), "prod");
}

TEST(CBindingsTypes, ReduceOpNameMin) {
    EXPECT_STREQ(dtl_reduce_op_name(DTL_OP_MIN), "min");
}

TEST(CBindingsTypes, ReduceOpNameMax) {
    EXPECT_STREQ(dtl_reduce_op_name(DTL_OP_MAX), "max");
}

TEST(CBindingsTypes, ReduceOpNameInvalidReturnsUnknown) {
    EXPECT_STREQ(dtl_reduce_op_name((dtl_reduce_op)999), "unknown");
}

// ============================================================================
// Shape Tests - 1D
// ============================================================================

TEST(CBindingsTypes, Shape1dNdim) {
    dtl_shape shape = dtl_shape_1d(100);
    EXPECT_EQ(shape.ndim, 1);
}

TEST(CBindingsTypes, Shape1dDim0) {
    dtl_shape shape = dtl_shape_1d(100);
    EXPECT_EQ(shape.dims[0], 100u);
}

TEST(CBindingsTypes, Shape1dSize) {
    dtl_shape shape = dtl_shape_1d(100);
    EXPECT_EQ(dtl_shape_size(&shape), 100u);
}

// ============================================================================
// Shape Tests - 2D
// ============================================================================

TEST(CBindingsTypes, Shape2dNdim) {
    dtl_shape shape = dtl_shape_2d(10, 20);
    EXPECT_EQ(shape.ndim, 2);
}

TEST(CBindingsTypes, Shape2dDims) {
    dtl_shape shape = dtl_shape_2d(10, 20);
    EXPECT_EQ(shape.dims[0], 10u);
    EXPECT_EQ(shape.dims[1], 20u);
}

TEST(CBindingsTypes, Shape2dSize) {
    dtl_shape shape = dtl_shape_2d(10, 20);
    EXPECT_EQ(dtl_shape_size(&shape), 200u);
}

// ============================================================================
// Shape Tests - 3D
// ============================================================================

TEST(CBindingsTypes, Shape3dNdim) {
    dtl_shape shape = dtl_shape_3d(2, 3, 4);
    EXPECT_EQ(shape.ndim, 3);
}

TEST(CBindingsTypes, Shape3dDims) {
    dtl_shape shape = dtl_shape_3d(2, 3, 4);
    EXPECT_EQ(shape.dims[0], 2u);
    EXPECT_EQ(shape.dims[1], 3u);
    EXPECT_EQ(shape.dims[2], 4u);
}

TEST(CBindingsTypes, Shape3dSize) {
    dtl_shape shape = dtl_shape_3d(2, 3, 4);
    EXPECT_EQ(dtl_shape_size(&shape), 24u);
}

// ============================================================================
// Shape Tests - ND
// ============================================================================

TEST(CBindingsTypes, ShapeNdCreatesCorrectly) {
    dtl_size_t dims[4] = {2, 3, 4, 5};
    dtl_shape shape = dtl_shape_nd(4, dims);
    EXPECT_EQ(shape.ndim, 4);
    EXPECT_EQ(shape.dims[0], 2u);
    EXPECT_EQ(shape.dims[1], 3u);
    EXPECT_EQ(shape.dims[2], 4u);
    EXPECT_EQ(shape.dims[3], 5u);
}

TEST(CBindingsTypes, ShapeNdSize) {
    dtl_size_t dims[4] = {2, 3, 4, 5};
    dtl_shape shape = dtl_shape_nd(4, dims);
    EXPECT_EQ(dtl_shape_size(&shape), 120u);
}

TEST(CBindingsTypes, ShapeNdClampedToMax) {
    dtl_size_t dims[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    dtl_shape shape = dtl_shape_nd(10, dims);
    EXPECT_EQ(shape.ndim, DTL_MAX_TENSOR_RANK);  // Clamped
}

TEST(CBindingsTypes, ShapeNdWithNullDims) {
    dtl_shape shape = dtl_shape_nd(3, nullptr);
    EXPECT_EQ(shape.ndim, 3);
    EXPECT_EQ(shape.dims[0], 0u);
    EXPECT_EQ(shape.dims[1], 0u);
    EXPECT_EQ(shape.dims[2], 0u);
}

TEST(CBindingsTypes, ShapeNdNegativeNdim) {
    dtl_size_t dims[3] = {1, 2, 3};
    dtl_shape shape = dtl_shape_nd(-1, dims);
    EXPECT_EQ(shape.ndim, 0);
}

// ============================================================================
// Shape Size Edge Cases
// ============================================================================

TEST(CBindingsTypes, ShapeSizeNullReturnsZero) {
    EXPECT_EQ(dtl_shape_size(nullptr), 0u);
}

TEST(CBindingsTypes, ShapeSizeZeroNdimReturnsZero) {
    dtl_shape shape;
    shape.ndim = 0;
    EXPECT_EQ(dtl_shape_size(&shape), 0u);
}

TEST(CBindingsTypes, ShapeSizeNegativeNdimReturnsZero) {
    dtl_shape shape;
    shape.ndim = -1;
    EXPECT_EQ(dtl_shape_size(&shape), 0u);
}
