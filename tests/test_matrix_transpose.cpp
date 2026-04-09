#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/concurrent_blocked_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"

#include <gtest/gtest.h>
#include <memory>

class MatrixTransposeTest : public testing::TestWithParam<std::shared_ptr<TensorStrategy>> {};

TEST_P(MatrixTransposeTest, Transpose) {

    auto strategy = GetParam();
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, strategy);

    Tensor b = a.transpose();

    ASSERT_EQ(b.getWidth(), 2);
    ASSERT_EQ(b.getHeight(), 2);
    ASSERT_EQ(b.data[0], 1.0f);
    ASSERT_EQ(b.data[1], 3.0f);
    ASSERT_EQ(b.data[2], 2.0f);
    ASSERT_EQ(b.data[3], 4.0f);
}

INSTANTIATE_TEST_SUITE_P(
    TestAllTransposeStrategies,
    MatrixTransposeTest,
    testing::Values(
        std::make_shared<BasicTensorStrategy>(),
        std::make_shared<SimdTensorStrategy>(),
        std::make_shared<ConcurrentRowTensorStrategy>(),
        std::make_shared<ConcurrentBlockedTensorStrategy>(),
        std::make_shared<OptimisedTensorStrategy>()
    )
);




