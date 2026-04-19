#include <gtest/gtest.h>
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

class TestScalarOps : public testing::TestWithParam<TensorStrategy*> {};

TEST_P(TestScalarOps, TestUnaryMinus) {

    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());
    Tensor b = -a;

    ASSERT_FLOAT_EQ(b.data[0], -1.0f);
    ASSERT_FLOAT_EQ(b.data[1], -2.0f);
    ASSERT_FLOAT_EQ(b.data[2], -3.0f);
    ASSERT_FLOAT_EQ(b.data[3], -4.0f);
}

TEST_P(TestScalarOps, TestScalarMult) {

    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());
    Tensor b = a * 2.0f;

    ASSERT_FLOAT_EQ(b.data[0], 2.0f);
    ASSERT_FLOAT_EQ(b.data[1], 4.0f);
    ASSERT_FLOAT_EQ(b.data[2], 6.0f);
    ASSERT_FLOAT_EQ(b.data[3], 8.0f);
}

TEST_P(TestScalarOps, TestScalarDiv) {

    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());
    Tensor b = a / 2.0f;

    ASSERT_FLOAT_EQ(b.data[0], 0.5f);
    ASSERT_FLOAT_EQ(b.data[1], 1.0f);
    ASSERT_FLOAT_EQ(b.data[2], 1.5f);
    ASSERT_FLOAT_EQ(b.data[3], 2.0f);
}

INSTANTIATE_TEST_CASE_P(TensorStrategies, TestScalarOps, testing::Values(
    &BasicTensorStrategy::get_instance(),
    &SimdTensorStrategy::get_instance(),
    &ConcurrentRowTensorStrategy::get_instance(),
    &OptimisedTensorStrategy::get_instance(),
    &BlockedSimdTensorStrategy::get_instance()
));
