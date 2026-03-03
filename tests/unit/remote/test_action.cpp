// Copyright (c) 2026 Bryce M. Westheimer
// SPDX-License-Identifier: BSD-3-Clause

/// @file test_action.cpp
/// @brief Unit tests for dtl/remote/action.hpp
/// @details Tests action registration, ID computation, and traits.

#include <dtl/remote/action.hpp>
#include <dtl/remote/action_registry.hpp>

#include <gtest/gtest.h>

#include <string>

namespace dtl::remote::test {

// =============================================================================
// Test Functions
// =============================================================================

int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
void notify(int x) { (void)x; }
std::string concat(const std::string& a, const std::string& b) { return a + b; }
int unary(int x) { return x * 2; }
int nullary() { return 42; }

// Register actions
DTL_REGISTER_ACTION(add);
DTL_REGISTER_ACTION(subtract);
DTL_REGISTER_ACTION(notify);
DTL_REGISTER_ACTION(concat);
DTL_REGISTER_ACTION(unary);
DTL_REGISTER_ACTION(nullary);

// =============================================================================
// Action ID Tests
// =============================================================================

TEST(ActionIdTest, ActionIdNonZero) {
    EXPECT_NE(action_add.id(), 0u);
    EXPECT_NE(action_subtract.id(), 0u);
    EXPECT_NE(action_notify.id(), 0u);
}

TEST(ActionIdTest, DifferentFunctionsHaveDifferentIds) {
    // Different functions should have different IDs
    EXPECT_NE(action_add.id(), action_subtract.id());
    EXPECT_NE(action_add.id(), action_unary.id());
    EXPECT_NE(action_subtract.id(), action_notify.id());
}

TEST(ActionIdTest, SameActionHasSameId) {
    // Same action should always return same ID
    EXPECT_EQ(action_add.id(), action<&add>::id());
    EXPECT_EQ(action_subtract.id(), action<&subtract>::id());
}

TEST(ActionIdTest, ActionIdIsNonZero) {
    // Verify action IDs are non-zero and unique
    auto add_id = action<&add>::id();
    auto sub_id = action<&subtract>::id();

    EXPECT_NE(add_id, 0u) << "Action ID should be non-zero";
    EXPECT_NE(add_id, sub_id) << "Different functions should have different IDs";
}

// =============================================================================
// Action Type Traits Tests
// =============================================================================

TEST(ActionTraitsTest, RequestType) {
    // add(int, int) -> request is tuple<int, int>
    using add_req = typename action<&add>::request_type;
    static_assert(std::is_same_v<add_req, std::tuple<int, int>>);

    // unary(int) -> request is tuple<int>
    using unary_req = typename action<&unary>::request_type;
    static_assert(std::is_same_v<unary_req, std::tuple<int>>);

    // nullary() -> request is tuple<>
    using null_req = typename action<&nullary>::request_type;
    static_assert(std::is_same_v<null_req, std::tuple<>>);
}

TEST(ActionTraitsTest, ResponseType) {
    // add returns int
    using add_resp = typename action<&add>::response_type;
    static_assert(std::is_same_v<add_resp, int>);

    // notify returns void
    using notify_resp = typename action<&notify>::response_type;
    static_assert(std::is_void_v<notify_resp>);

    // concat returns string
    using concat_resp = typename action<&concat>::response_type;
    static_assert(std::is_same_v<concat_resp, std::string>);
}

TEST(ActionTraitsTest, Arity) {
    static_assert(action<&add>::arity == 2);
    static_assert(action<&unary>::arity == 1);
    static_assert(action<&nullary>::arity == 0);
    static_assert(action<&notify>::arity == 1);
}

TEST(ActionTraitsTest, IsVoid) {
    static_assert(!action<&add>::is_void);
    static_assert(action<&notify>::is_void);
    static_assert(!action<&nullary>::is_void);
}

// =============================================================================
// Action Invocation Tests
// =============================================================================

TEST(ActionInvokeTest, InvokeDirectly) {
    EXPECT_EQ(action<&add>::invoke(10, 20), 30);
    EXPECT_EQ(action<&subtract>::invoke(50, 30), 20);
    EXPECT_EQ(action<&unary>::invoke(5), 10);
    EXPECT_EQ(action<&nullary>::invoke(), 42);
}

TEST(ActionInvokeTest, InvokeWithTuple) {
    std::tuple<int, int> args{3, 4};
    EXPECT_EQ(action<&add>::invoke_tuple(args), 7);

    std::tuple<int> unary_args{10};
    EXPECT_EQ(action<&unary>::invoke_tuple(unary_args), 20);

    std::tuple<> nullary_args{};
    EXPECT_EQ(action<&nullary>::invoke_tuple(nullary_args), 42);
}

TEST(ActionInvokeTest, GetFunction) {
    auto fn = action<&add>::function();
    EXPECT_EQ(fn(100, 200), 300);
}

// =============================================================================
// is_action Trait Tests
// =============================================================================

TEST(IsActionTest, ActionTypesAreDetected) {
    static_assert(is_action_v<action<&add>>);
    static_assert(is_action_v<action<&notify>>);
    static_assert(is_action_v<decltype(action_add)>);
}

TEST(IsActionTest, NonActionTypesAreNotDetected) {
    static_assert(!is_action_v<int>);
    static_assert(!is_action_v<std::string>);
    static_assert(!is_action_v<void>);
}

// =============================================================================
// Action Concept Tests
// =============================================================================

TEST(ActionConceptTest, ConceptSatisfied) {
    static_assert(Action<action<&add>>);
    static_assert(Action<action<&notify>>);
}

// =============================================================================
// Action List Tests
// =============================================================================

TEST(ActionListTest, Contains) {
    using list = action_list<action<&add>, action<&subtract>>;

    EXPECT_TRUE(list::contains(action<&add>::id()));
    EXPECT_TRUE(list::contains(action<&subtract>::id()));
    EXPECT_FALSE(list::contains(action<&notify>::id()));
}

TEST(ActionListTest, IndexOf) {
    using list = action_list<action<&add>, action<&subtract>, action<&notify>>;

    EXPECT_EQ(list::index_of(action<&add>::id()), 0);
    EXPECT_EQ(list::index_of(action<&subtract>::id()), 1);
    EXPECT_EQ(list::index_of(action<&notify>::id()), 2);
    EXPECT_EQ(list::index_of(action<&unary>::id()), -1);
}

TEST(ActionListTest, Size) {
    using list = action_list<action<&add>, action<&subtract>>;
    static_assert(list::size == 2);

    using empty_list = action_list<>;
    static_assert(empty_list::size == 0);
}

// =============================================================================
// Static Action Table Tests
// =============================================================================

TEST(StaticActionTableTest, Dispatch) {
    using table = static_action_table<action<&add>, action<&subtract>>;

    bool found = false;
    bool result = table::dispatch(action<&add>::id(), [&](auto act) {
        found = true;
        using A = decltype(act);
        EXPECT_EQ(A::invoke(5, 3), 8);
    });

    EXPECT_TRUE(result);
    EXPECT_TRUE(found);
}

TEST(StaticActionTableTest, DispatchNotFound) {
    using table = static_action_table<action<&add>>;

    bool result = table::dispatch(action<&subtract>::id(), [](auto) {});
    EXPECT_FALSE(result);
}

}  // namespace dtl::remote::test
