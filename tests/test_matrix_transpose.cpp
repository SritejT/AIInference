#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

#include <gtest/gtest.h>

class MatrixTransposeTest : public testing::TestWithParam<TensorStrategy*> {};

TEST_P(MatrixTransposeTest, TransposeSmallMatrix) {

    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());

    Tensor b = a.transpose();

    ASSERT_EQ(b.getWidth(), 2);
    ASSERT_EQ(b.getHeight(), 2);
    ASSERT_EQ(b.data[0], 1.0f);
    ASSERT_EQ(b.data[1], 3.0f);
    ASSERT_EQ(b.data[2], 2.0f);
    ASSERT_EQ(b.data[3], 4.0f);
}

TEST_P(MatrixTransposeTest, TransposeLargeMatrix) {

    std::vector<float> data(1000000);

    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            data[i * 1000 + j] = i;
        }
    }

    Tensor a = Tensor(data, 1000, 1000, GetParam());

    Tensor b = a.transpose();

    ASSERT_EQ(b.getWidth(), 1000);
    ASSERT_EQ(b.getHeight(), 1000);

    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 1000; j++) {
            ASSERT_EQ(b.data[j * 1000 + i], i);
        }
    }
    
}

INSTANTIATE_TEST_SUITE_P(TestAllTransposeStrategies, MatrixTransposeTest, testing::Values(
    &BasicTensorStrategy::get_instance(),
    &SimdTensorStrategy::get_instance(),
    &ConcurrentRowTensorStrategy::get_instance(),
    &OptimisedTensorStrategy::get_instance(),
    &BlockedSimdTensorStrategy::get_instance()
));




