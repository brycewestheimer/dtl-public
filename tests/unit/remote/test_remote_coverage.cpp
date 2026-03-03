// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_remote_coverage.cpp
/// @brief Unit tests for remote module: rma_remote_ref, action types
/// @details Phase 14 T05: rma_remote_ref construction, get/set, locality checks,
///          const specialization, factory functions, action compile-time metadata,
///          type traits, registration macros.
/// @note rpc.hpp is excluded due to upstream compilation issues in
///       rpc_serialization.hpp (missing declarations, type errors).

#include <dtl/rma/remote_integration.hpp>
#include <dtl/communication/memory_window.hpp>
#include <dtl/error/result.hpp>
#include <dtl/remote/action.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace dtl::test {

// =============================================================================
// Helper: create a test window with known data
// =============================================================================

static memory_window make_test_window(size_type sz = 1024) {
    auto res = memory_window::allocate(sz);
    EXPECT_TRUE(res.has_value());
    auto win = std::move(res.value());
    if (win.base()) {
        std::memset(win.base(), 0, sz);
    }
    return win;
}

// =============================================================================
// rma_remote_ref Construction Tests
// =============================================================================

TEST(RmaRemoteRefTest, DefaultConstructorInvalid) {
    rma::rma_remote_ref<int> ref;
    EXPECT_FALSE(ref.valid());
}

TEST(RmaRemoteRefTest, ConstructWithWindow) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(0, 0, win, 0);
    EXPECT_TRUE(ref.valid());
    EXPECT_EQ(ref.owner_rank(), 0);
    EXPECT_EQ(ref.offset(), 0u);
}

TEST(RmaRemoteRefTest, ConstructWithOffset) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(0, sizeof(int), win, 0);
    EXPECT_EQ(ref.offset(), sizeof(int));
}

TEST(RmaRemoteRefTest, ConstructWithNonZeroOwner) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(5, 0, win, 0);
    EXPECT_EQ(ref.owner_rank(), 5);
    EXPECT_TRUE(ref.valid());
}

// =============================================================================
// rma_remote_ref Locality Tests
// =============================================================================

TEST(RmaRemoteRefTest, IsLocalWhenOwnerEqualsRank) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(0, 0, win, 0);
    EXPECT_TRUE(ref.is_local());
    EXPECT_FALSE(ref.is_remote());
}

TEST(RmaRemoteRefTest, IsRemoteWhenOwnerDiffers) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(1, 0, win, 0);
    EXPECT_TRUE(ref.is_remote());
    EXPECT_FALSE(ref.is_local());
}

TEST(RmaRemoteRefTest, IsLocalWithNonZeroLocalRank) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(3, 0, win, 3);
    EXPECT_TRUE(ref.is_local());
}

// =============================================================================
// rma_remote_ref Get Tests (single-process, local only)
// =============================================================================

TEST(RmaRemoteRefTest, GetLocalValue) {
    auto win = make_test_window(256);
    int expected = 42;
    std::memcpy(win.base(), &expected, sizeof(int));

    rma::rma_remote_ref<int> ref(0, 0, win, 0);
    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 42);
}

TEST(RmaRemoteRefTest, GetAtOffset) {
    auto win = make_test_window(256);
    int val0 = 10, val1 = 20;
    std::memcpy(win.base(), &val0, sizeof(int));
    std::memcpy(static_cast<char*>(win.base()) + sizeof(int), &val1, sizeof(int));

    rma::rma_remote_ref<int> ref(0, sizeof(int), win, 0);
    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 20);
}

TEST(RmaRemoteRefTest, GetOnInvalidRefFails) {
    rma::rma_remote_ref<int> ref;
    auto res = ref.get();
    EXPECT_TRUE(res.has_error());
}

// =============================================================================
// rma_remote_ref Put Tests (single-process, local only)
// =============================================================================

TEST(RmaRemoteRefTest, PutLocalValue) {
    auto win = make_test_window(256);
    rma::rma_remote_ref<int> ref(0, 0, win, 0);

    auto res = ref.put(99);
    ASSERT_TRUE(res.has_value());

    int actual = 0;
    std::memcpy(&actual, win.base(), sizeof(int));
    EXPECT_EQ(actual, 99);
}

TEST(RmaRemoteRefTest, PutThenGet) {
    auto win = make_test_window(256);
    rma::rma_remote_ref<int> ref(0, 0, win, 0);

    auto put_res = ref.put(77);
    ASSERT_TRUE(put_res.has_value());

    auto get_res = ref.get();
    ASSERT_TRUE(get_res.has_value());
    EXPECT_EQ(get_res.value(), 77);
}

TEST(RmaRemoteRefTest, PutOnInvalidRefFails) {
    rma::rma_remote_ref<int> ref;
    auto res = ref.put(42);
    EXPECT_TRUE(res.has_error());
}

TEST(RmaRemoteRefTest, PutMoveValue) {
    auto win = make_test_window(256);
    rma::rma_remote_ref<int> ref(0, 0, win, 0);
    int val = 55;
    auto res = ref.put(std::move(val));
    ASSERT_TRUE(res.has_value());

    auto get_res = ref.get();
    ASSERT_TRUE(get_res.has_value());
    EXPECT_EQ(get_res.value(), 55);
}

TEST(RmaRemoteRefTest, PutMultipleValues) {
    auto win = make_test_window(256);
    rma::rma_remote_ref<int> ref0(0, 0, win, 0);
    rma::rma_remote_ref<int> ref1(0, sizeof(int), win, 0);

    ASSERT_TRUE(ref0.put(100).has_value());
    ASSERT_TRUE(ref1.put(200).has_value());

    EXPECT_EQ(ref0.get().value(), 100);
    EXPECT_EQ(ref1.get().value(), 200);
}

// =============================================================================
// rma_remote_ref Window Access
// =============================================================================

TEST(RmaRemoteRefTest, WindowPointerAccess) {
    auto win = make_test_window();
    rma::rma_remote_ref<int> ref(0, 0, win, 0);
    EXPECT_NE(ref.window(), nullptr);
}

TEST(RmaRemoteRefTest, DefaultRefWindowIsNull) {
    rma::rma_remote_ref<int> ref;
    EXPECT_EQ(ref.window(), nullptr);
}

// =============================================================================
// rma_remote_ref Const Specialization Tests
// =============================================================================

TEST(RmaRemoteRefConstTest, ConstructValid) {
    auto win = make_test_window(256);
    rma::rma_remote_ref<const int> ref(0, 0, win, 0);
    EXPECT_TRUE(ref.valid());
}

TEST(RmaRemoteRefConstTest, GetWorks) {
    auto win = make_test_window(256);
    int expected = 123;
    std::memcpy(win.base(), &expected, sizeof(int));

    rma::rma_remote_ref<const int> ref(0, 0, win, 0);
    auto res = ref.get();
    ASSERT_TRUE(res.has_value());
    EXPECT_EQ(res.value(), 123);
}

TEST(RmaRemoteRefConstTest, LocalAndRemoteChecks) {
    auto win = make_test_window();
    rma::rma_remote_ref<const int> local_ref(0, 0, win, 0);
    EXPECT_TRUE(local_ref.is_local());
    EXPECT_FALSE(local_ref.is_remote());

    rma::rma_remote_ref<const int> remote_ref(1, 0, win, 0);
    EXPECT_TRUE(remote_ref.is_remote());
    EXPECT_FALSE(remote_ref.is_local());
}

TEST(RmaRemoteRefConstTest, DefaultConstructorInvalid) {
    rma::rma_remote_ref<const int> ref;
    EXPECT_FALSE(ref.valid());
}

TEST(RmaRemoteRefConstTest, WindowPointer) {
    auto win = make_test_window();
    rma::rma_remote_ref<const int> ref(0, 0, win, 0);
    EXPECT_NE(ref.window(), nullptr);

    rma::rma_remote_ref<const int> def;
    EXPECT_EQ(def.window(), nullptr);
}

TEST(RmaRemoteRefConstTest, OffsetAndOwner) {
    auto win = make_test_window();
    rma::rma_remote_ref<const int> ref(2, 64, win, 0);
    EXPECT_EQ(ref.owner_rank(), 2);
    EXPECT_EQ(ref.offset(), 64u);
}

// =============================================================================
// Factory Function Tests
// =============================================================================

TEST(RmaRemoteRefFactoryTest, MakeRmaRef) {
    auto win = make_test_window();
    auto ref = rma::make_rma_ref<int>(0, 0, win);
    EXPECT_TRUE(ref.valid());
    EXPECT_EQ(ref.owner_rank(), 0);
    EXPECT_EQ(ref.offset(), 0u);
}

TEST(RmaRemoteRefFactoryTest, MakeRmaRefWithLocalRank) {
    auto win = make_test_window();
    auto ref = rma::make_rma_ref<int>(0, 0, win, 0);
    EXPECT_TRUE(ref.is_local());
}

TEST(RmaRemoteRefFactoryTest, MakeRmaRefIndexed) {
    auto win = make_test_window();
    auto ref = rma::make_rma_ref_indexed<int>(0, 3, win);
    EXPECT_EQ(ref.offset(), 3 * sizeof(int));
}

TEST(RmaRemoteRefFactoryTest, MakeRmaRefIndexedZero) {
    auto win = make_test_window();
    auto ref = rma::make_rma_ref_indexed<int>(0, 0, win);
    EXPECT_EQ(ref.offset(), 0u);
}

TEST(RmaRemoteRefFactoryTest, MakeRmaRefWithByteOffset) {
    auto win = make_test_window();
    auto ref = rma::make_rma_ref<double>(0, 128, win);
    EXPECT_EQ(ref.offset(), 128u);
}

// =============================================================================
// rma_remote_ref Type Trait Tests
// =============================================================================

TEST(RmaRemoteRefTraitTest, IsRmaRemoteRefTrue) {
    static_assert(rma::is_rma_remote_ref_v<rma::rma_remote_ref<int>>);
    static_assert(rma::is_rma_remote_ref_v<rma::rma_remote_ref<double>>);
    static_assert(rma::is_rma_remote_ref_v<rma::rma_remote_ref<const int>>);
    SUCCEED();
}

TEST(RmaRemoteRefTraitTest, IsRmaRemoteRefFalse) {
    static_assert(!rma::is_rma_remote_ref_v<int>);
    static_assert(!rma::is_rma_remote_ref_v<std::string>);
    static_assert(!rma::is_rma_remote_ref_v<memory_window>);
    SUCCEED();
}

// =============================================================================
// Action Type Tests
// =============================================================================

namespace {
int test_add(int a, int b) { return a + b; }
void test_noop() {}
double test_double(double x) { return x * 2.0; }
std::string test_greet(int n) { return "hello-" + std::to_string(n); }
}  // namespace

TEST(ActionTest, ActionIdNonZero) {
    auto id = remote::action<&test_add>::id();
    EXPECT_NE(id, remote::invalid_action_id);
}

TEST(ActionTest, ActionMetadata) {
    using Add = remote::action<&test_add>;
    static_assert(Add::arity == 2);
    static_assert(!Add::is_void);
    static_assert(std::is_same_v<Add::response_type, int>);
    SUCCEED();
}

TEST(ActionTest, VoidActionMetadata) {
    using Noop = remote::action<&test_noop>;
    static_assert(Noop::arity == 0);
    static_assert(Noop::is_void);
    SUCCEED();
}

TEST(ActionTest, InvokeDirectly) {
    auto val = remote::action<&test_add>::invoke(3, 4);
    EXPECT_EQ(val, 7);
}

TEST(ActionTest, DifferentActionsHaveDifferentIds) {
    auto id1 = remote::action<&test_add>::id();
    auto id2 = remote::action<&test_noop>::id();
    EXPECT_NE(id1, id2);
}

TEST(ActionTest, InvokeDoubleAction) {
    auto val = remote::action<&test_double>::invoke(3.5);
    EXPECT_DOUBLE_EQ(val, 7.0);
}

TEST(ActionTest, InvokeStringAction) {
    auto val = remote::action<&test_greet>::invoke(42);
    EXPECT_EQ(val, "hello-42");
}

TEST(ActionTest, ActionIdIsStable) {
    auto id_a = remote::action<&test_add>::id();
    auto id_b = remote::action<&test_add>::id();
    EXPECT_EQ(id_a, id_b);
}

TEST(ActionTest, FunctionPointerAccessible) {
    auto fn = remote::action<&test_add>::function();
    EXPECT_EQ(fn(10, 20), 30);
}

TEST(ActionTest, InvokeTuple) {
    using Add = remote::action<&test_add>;
    Add::request_type args{5, 6};
    auto val = Add::invoke_tuple(args);
    EXPECT_EQ(val, 11);
}

TEST(ActionTest, RequestTypeIsArgsTuple) {
    using Add = remote::action<&test_add>;
    static_assert(std::is_same_v<Add::request_type, std::tuple<int, int>>);
    SUCCEED();
}

TEST(ActionTest, SingleArgAction) {
    using Dbl = remote::action<&test_double>;
    static_assert(Dbl::arity == 1);
    static_assert(!Dbl::is_void);
    static_assert(std::is_same_v<Dbl::response_type, double>);
    static_assert(std::is_same_v<Dbl::request_type, std::tuple<double>>);
    SUCCEED();
}

// =============================================================================
// Action Trait Tests
// =============================================================================

TEST(ActionTraitTest, IsActionTrue) {
    static_assert(remote::is_action_v<remote::action<&test_add>>);
    static_assert(remote::is_action_v<remote::action<&test_noop>>);
    SUCCEED();
}

TEST(ActionTraitTest, IsActionFalse) {
    static_assert(!remote::is_action_v<int>);
    static_assert(!remote::is_action_v<double>);
    static_assert(!remote::is_action_v<std::string>);
    SUCCEED();
}

TEST(ActionTraitTest, ActionConceptSatisfied) {
    static_assert(remote::Action<remote::action<&test_add>>);
    static_assert(remote::Action<remote::action<&test_noop>>);
    SUCCEED();
}

TEST(ActionTraitTest, ActionConceptNotSatisfied) {
    static_assert(!remote::Action<int>);
    static_assert(!remote::Action<void>);
    SUCCEED();
}

TEST(ActionTraitTest, GetActionIdFreeFunction) {
    auto id = remote::get_action_id<remote::action<&test_add>>();
    EXPECT_NE(id, remote::invalid_action_id);
    EXPECT_EQ(id, remote::action<&test_add>::id());
}

TEST(ActionTraitTest, GetActionIdFromInstance) {
    remote::action<&test_add> act{};
    auto id = remote::get_action_id(act);
    EXPECT_EQ(id, remote::action<&test_add>::id());
}

// =============================================================================
// Registration Macro Tests
// =============================================================================

namespace {
int registered_fn(int x) { return x * 3; }
}  // namespace

DTL_REGISTER_ACTION(registered_fn);

TEST(ActionRegistrationTest, RegisteredActionHasId) {
    auto id = action_registered_fn.id();
    EXPECT_NE(id, remote::invalid_action_id);
}

TEST(ActionRegistrationTest, RegisteredActionInvoke) {
    auto val = decltype(action_registered_fn)::invoke(7);
    EXPECT_EQ(val, 21);
}

// =============================================================================
// Invalid Action ID Sentinel
// =============================================================================

TEST(ActionConstantTest, InvalidActionIdIsZero) {
    EXPECT_EQ(remote::invalid_action_id, 0u);
}

}  // namespace dtl::test
