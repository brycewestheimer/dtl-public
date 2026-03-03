// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_dynamic_handler_local.cpp
/// @brief Unit tests for dynamic RPC handler creation and local invocation
/// @details Tests that dynamic handlers correctly serialize/deserialize args
///          and invoke functions.

#include <dtl/remote/action.hpp>
#include <dtl/remote/action_registry.hpp>
#include <dtl/remote/dynamic_handler.hpp>
#include <dtl/remote/argument_pack.hpp>
#include <dtl/serialization/serializer.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <string>

namespace dtl::remote::test {

// =============================================================================
// Test Functions
// =============================================================================

int add_ints(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
int identity(int x) { return x; }
int constant() { return 42; }
void no_op() {}
void sink(int x) { (void)x; }

double add_doubles(double a, double b) { return a + b; }
float add_floats(float a, float b) { return a + b; }

// Struct for testing POD serialization
struct Point {
    int x;
    int y;
};

Point add_points(Point a, Point b) {
    return Point{a.x + b.x, a.y + b.y};
}

int point_magnitude_squared(Point p) {
    return p.x * p.x + p.y * p.y;
}

// Register actions
DTL_REGISTER_ACTION(add_ints);
DTL_REGISTER_ACTION(multiply);
DTL_REGISTER_ACTION(identity);
DTL_REGISTER_ACTION(constant);
DTL_REGISTER_ACTION(no_op);
DTL_REGISTER_ACTION(sink);
DTL_REGISTER_ACTION(add_doubles);
DTL_REGISTER_ACTION(add_floats);
DTL_REGISTER_ACTION(add_points);
DTL_REGISTER_ACTION(point_magnitude_squared);

// =============================================================================
// Basic Handler Creation Tests
// =============================================================================

TEST(DynamicHandlerTest, MakeHandlerReturnsValidHandler) {
    // Create registry with dynamic handlers
    auto registry = registry_builder<16>{}
        .add<action<&add_ints>>()
        .add<action<&multiply>>()
        .build();
    
    // Handlers should be valid (not null)
    auto add_handler = registry.find(action_add_ints.id());
    ASSERT_TRUE(add_handler.has_value());
    EXPECT_TRUE(add_handler->valid());
    
    auto mul_handler = registry.find(action_multiply.id());
    ASSERT_TRUE(mul_handler.has_value());
    EXPECT_TRUE(mul_handler->valid());
}

TEST(DynamicHandlerTest, UnknownActionNotFound) {
    auto registry = registry_builder<16>{}
        .add<action<&add_ints>>()
        .build();
    
    // Unknown action ID
    auto unknown = registry.find(action_multiply.id());
    EXPECT_FALSE(unknown.has_value());
}

// =============================================================================
// Local Invocation Tests
// =============================================================================

TEST(DynamicHandlerTest, InvokeAddReturnsCorrectResult) {
    auto registry = registry_builder<16>{}
        .add<action<&add_ints>>()
        .build();
    
    auto handler = registry.find(action_add_ints.id());
    ASSERT_TRUE(handler.has_value());
    ASSERT_TRUE(handler->valid());
    
    // Serialize arguments: add(10, 32)
    std::array<std::byte, 64> request_buf{};
    argument_pack<int, int>::serialize(10, 32, request_buf.data());
    
    // Invoke
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(int) * 2,
        response_buf.data(), response_buf.size());
    
    // Deserialize result
    ASSERT_EQ(written, sizeof(int));
    int result = deserialize<int>(response_buf.data(), written);
    EXPECT_EQ(result, 42);
}

TEST(DynamicHandlerTest, InvokeMultiplyReturnsCorrectResult) {
    auto registry = registry_builder<16>{}
        .add<action<&multiply>>()
        .build();
    
    auto handler = registry.find(action_multiply.id());
    ASSERT_TRUE(handler.has_value());
    
    // multiply(7, 6)
    std::array<std::byte, 64> request_buf{};
    argument_pack<int, int>::serialize(7, 6, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(int) * 2,
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(int));
    int result = deserialize<int>(response_buf.data(), written);
    EXPECT_EQ(result, 42);
}

TEST(DynamicHandlerTest, InvokeUnaryFunction) {
    auto registry = registry_builder<16>{}
        .add<action<&identity>>()
        .build();
    
    auto handler = registry.find(action_identity.id());
    ASSERT_TRUE(handler.has_value());
    
    // identity(123)
    std::array<std::byte, 64> request_buf{};
    argument_pack<int>::serialize(123, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(int),
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(int));
    int result = deserialize<int>(response_buf.data(), written);
    EXPECT_EQ(result, 123);
}

TEST(DynamicHandlerTest, InvokeNullaryFunction) {
    auto registry = registry_builder<16>{}
        .add<action<&constant>>()
        .build();
    
    auto handler = registry.find(action_constant.id());
    ASSERT_TRUE(handler.has_value());
    
    // constant() - no arguments
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        nullptr, 0,
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(int));
    int result = deserialize<int>(response_buf.data(), written);
    EXPECT_EQ(result, 42);
}

TEST(DynamicHandlerTest, InvokeVoidFunctionReturnsZeroBytes) {
    auto registry = registry_builder<16>{}
        .add<action<&no_op>>()
        .add<action<&sink>>()
        .build();
    
    // no_op()
    auto no_op_handler = registry.find(action_no_op.id());
    ASSERT_TRUE(no_op_handler.has_value());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = no_op_handler->invoke(
        nullptr, 0,
        response_buf.data(), response_buf.size());
    EXPECT_EQ(written, 0u);  // Void return
    
    // sink(42)
    auto sink_handler = registry.find(action_sink.id());
    ASSERT_TRUE(sink_handler.has_value());
    
    std::array<std::byte, 64> request_buf{};
    argument_pack<int>::serialize(42, request_buf.data());
    
    written = sink_handler->invoke(
        request_buf.data(), sizeof(int),
        response_buf.data(), response_buf.size());
    EXPECT_EQ(written, 0u);  // Void return
}

// =============================================================================
// Different Type Tests
// =============================================================================

TEST(DynamicHandlerTest, InvokeWithDoubles) {
    auto registry = registry_builder<16>{}
        .add<action<&add_doubles>>()
        .build();
    
    auto handler = registry.find(action_add_doubles.id());
    ASSERT_TRUE(handler.has_value());
    
    // add_doubles(3.14, 2.71)
    std::array<std::byte, 64> request_buf{};
    argument_pack<double, double>::serialize(3.14, 2.71, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(double) * 2,
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(double));
    double result = deserialize<double>(response_buf.data(), written);
    EXPECT_DOUBLE_EQ(result, 5.85);
}

TEST(DynamicHandlerTest, InvokeWithFloats) {
    auto registry = registry_builder<16>{}
        .add<action<&add_floats>>()
        .build();
    
    auto handler = registry.find(action_add_floats.id());
    ASSERT_TRUE(handler.has_value());
    
    // add_floats(1.5f, 2.5f)
    std::array<std::byte, 64> request_buf{};
    argument_pack<float, float>::serialize(1.5f, 2.5f, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(float) * 2,
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(float));
    float result = deserialize<float>(response_buf.data(), written);
    EXPECT_FLOAT_EQ(result, 4.0f);
}

TEST(DynamicHandlerTest, InvokeWithPODStruct) {
    auto registry = registry_builder<16>{}
        .add<action<&add_points>>()
        .add<action<&point_magnitude_squared>>()
        .build();
    
    // add_points({1, 2}, {3, 4})
    auto add_handler = registry.find(action_add_points.id());
    ASSERT_TRUE(add_handler.has_value());
    
    std::array<std::byte, 64> request_buf{};
    Point p1{1, 2}, p2{3, 4};
    argument_pack<Point, Point>::serialize(p1, p2, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    size_type written = add_handler->invoke(
        request_buf.data(), sizeof(Point) * 2,
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(Point));
    Point result = deserialize<Point>(response_buf.data(), written);
    EXPECT_EQ(result.x, 4);
    EXPECT_EQ(result.y, 6);
    
    // point_magnitude_squared({3, 4}) = 25
    auto mag_handler = registry.find(action_point_magnitude_squared.id());
    ASSERT_TRUE(mag_handler.has_value());
    
    std::array<std::byte, 64> request_buf2{};
    Point p{3, 4};
    argument_pack<Point>::serialize(p, request_buf2.data());
    
    written = mag_handler->invoke(
        request_buf2.data(), sizeof(Point),
        response_buf.data(), response_buf.size());
    
    ASSERT_EQ(written, sizeof(int));
    int mag = deserialize<int>(response_buf.data(), written);
    EXPECT_EQ(mag, 25);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST(DynamicHandlerTest, InvalidHandlerReturnsZero) {
    action_handler invalid_handler;
    EXPECT_FALSE(invalid_handler.valid());
    
    std::array<std::byte, 64> buf{};
    size_type written = invalid_handler.invoke(
        buf.data(), 0, buf.data(), buf.size());
    EXPECT_EQ(written, 0u);
}

TEST(DynamicHandlerTest, InsufficientResponseBufferReturnsZero) {
    auto registry = registry_builder<16>{}
        .add<action<&add_ints>>()
        .build();
    
    auto handler = registry.find(action_add_ints.id());
    ASSERT_TRUE(handler.has_value());
    
    // Serialize arguments
    std::array<std::byte, 64> request_buf{};
    argument_pack<int, int>::serialize(10, 32, request_buf.data());
    
    // Response buffer too small for int result
    std::array<std::byte, 2> tiny_response{};
    size_type written = handler->invoke(
        request_buf.data(), sizeof(int) * 2,
        tiny_response.data(), tiny_response.size());
    
    EXPECT_EQ(written, 0u);  // Insufficient space
}

// =============================================================================
// Extended Handler Tests
// =============================================================================

TEST(ExtendedHandlerTest, CreateExtendedHandler) {
    auto handler = make_extended_handler<action<&add_ints>>();
    
    EXPECT_TRUE(handler.valid());
    EXPECT_EQ(handler.id(), action_add_ints.id());
    EXPECT_EQ(handler.arity(), 2u);
    EXPECT_FALSE(handler.is_void());
}

TEST(ExtendedHandlerTest, ExtendedHandlerVoidFunction) {
    auto handler = make_extended_handler<action<&no_op>>();
    
    EXPECT_TRUE(handler.valid());
    EXPECT_EQ(handler.arity(), 0u);
    EXPECT_TRUE(handler.is_void());
}

TEST(ExtendedHandlerTest, InvokeWithResultSuccess) {
    auto handler = make_extended_handler<action<&add_ints>>();
    
    std::array<std::byte, 64> request_buf{};
    argument_pack<int, int>::serialize(10, 32, request_buf.data());
    
    std::array<std::byte, 64> response_buf{};
    auto result = handler.invoke_with_result(
        request_buf.data(), sizeof(int) * 2,
        response_buf.data(), response_buf.size());
    
    EXPECT_TRUE(result.ok());
    EXPECT_EQ(result.bytes_written, sizeof(int));
    
    int value = deserialize<int>(response_buf.data(), result.bytes_written);
    EXPECT_EQ(value, 42);
}

TEST(ExtendedHandlerTest, InvokeWithResultBufferTooSmall) {
    auto handler = make_extended_handler<action<&add_ints>>();
    
    std::array<std::byte, 64> request_buf{};
    argument_pack<int, int>::serialize(10, 32, request_buf.data());
    
    std::array<std::byte, 2> tiny_response{};
    auto result = handler.invoke_with_result(
        request_buf.data(), sizeof(int) * 2,
        tiny_response.data(), tiny_response.size());
    
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.error_code, handler_result::err_buffer_too_small);
}

// =============================================================================
// Invoke Handler Utility Tests
// =============================================================================

TEST(InvokeHandlerTest, InvokeWithTypedArgs) {
    auto handler = make_dynamic_handler<action<&add_ints>>();
    
    auto result = invoke_handler<int>(handler, 10, 32);
    
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST(InvokeHandlerTest, InvokeVoidHandler) {
    auto handler = make_dynamic_handler<action<&no_op>>();
    
    auto result = invoke_handler<void>(handler);
    
    EXPECT_TRUE(result.has_value());
}

TEST(InvokeHandlerTest, InvokeInvalidHandlerReturnsError) {
    action_handler invalid;
    
    auto result = invoke_handler<int>(invalid, 1, 2);
    
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// Registry with Multiple Actions Tests
// =============================================================================

TEST(DynamicHandlerTest, RegistryWithMultipleActions) {
    auto registry = registry_builder<16>{}
        .add<action<&add_ints>>()
        .add<action<&multiply>>()
        .add<action<&identity>>()
        .add<action<&constant>>()
        .add<action<&no_op>>()
        .build();
    
    EXPECT_EQ(registry.size(), 5u);
    
    // All handlers should be valid
    EXPECT_TRUE(registry.find(action_add_ints.id())->valid());
    EXPECT_TRUE(registry.find(action_multiply.id())->valid());
    EXPECT_TRUE(registry.find(action_identity.id())->valid());
    EXPECT_TRUE(registry.find(action_constant.id())->valid());
    EXPECT_TRUE(registry.find(action_no_op.id())->valid());
}

TEST(DynamicHandlerTest, AddActionFromInstance) {
    auto registry = registry_builder<16>{}
        .add(action_add_ints)
        .add(action_multiply)
        .build();
    
    EXPECT_EQ(registry.size(), 2u);
    EXPECT_TRUE(registry.contains(action_add_ints.id()));
    EXPECT_TRUE(registry.contains(action_multiply.id()));
}

}  // namespace dtl::remote::test
