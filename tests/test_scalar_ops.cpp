#include <gtest/gtest.h>
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

class TestScalarOps : public testing::TestWithParam<std::shared_ptr<TensorStrategy>> {};

TEST_P(TestScalarOps, TestUnaryMinus) {
    auto strategy = GetParam();
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, strategy);
    Tensor b = -a;
    ASSERT_EQ(b.data[0], -1.0f);
    ASSERT_EQ(b.data[1], -2.0f);
    ASSERT_EQ(b.data[2], -3.0f);
    ASSERT_EQ(b.data[3], -4.0f);
}

TEST_P(TestScalarOps, TestScalarMult) {
    auto strategy = GetParam();
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, strategy);
    Tensor b = a * 2.0f;
    ASSERT_EQ(b.data[0], 2.0f);
    ASSERT_EQ(b.data[1], 4.0f);
    ASSERT_EQ(b.data[2], 6.0f);
    ASSERT_EQ(b.data[3], 8.0f);
}

TEST_P(TestScalarOps, TestScalarDiv) {
    auto strategy = GetParam();
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, strategy);
    Tensor b = a / 2.0f;
    ASSERT_EQ(b.data[0], 0.5f);
    ASSERT_EQ(b.data[1], 1.0f);
    ASSERT_EQ(b.data[2], 1.5f);
    ASSERT_EQ(b.data[3], 2.0f);
}

INSTANTIATE_TEST_CASE_P(TensorStrategies, TestScalarOps, testing::Values(
    std::make_shared<BasicTensorStrategy>(),
    std::make_shared<SimdTensorStrategy>(),
    std::make_shared<ConcurrentRowTensorStrategy>(),
    std::make_shared<OptimisedTensorStrategy>(),
    std::make_shared<BlockedSimdTensorStrategy>()
));
