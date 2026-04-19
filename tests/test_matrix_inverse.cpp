#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

#include <gtest/gtest.h>

class TestMatrixInverse : public testing::TestWithParam<TensorStrategy*> {};

TEST_P(TestMatrixInverse, CheckIdentityInverse) {

    Tensor A = Tensor({1.0f, 0.0f, 0.0f, 1.0f}, 2, 2, GetParam());

    Tensor Ainv = A.inverse();

    ASSERT_FLOAT_EQ(Ainv.data[0], 1.0f);
    ASSERT_FLOAT_EQ(Ainv.data[1], 0.0f);
    ASSERT_FLOAT_EQ(Ainv.data[2], 0.0f);
    ASSERT_FLOAT_EQ(Ainv.data[3], 1.0f);
}

TEST_P(TestMatrixInverse, Check2x2Inverse) {

    Tensor A = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());

    Tensor Ainv = A.inverse();

    ASSERT_FLOAT_EQ(Ainv.data[0], -2.0f);
    ASSERT_FLOAT_EQ(Ainv.data[1], 1.0f);
    ASSERT_FLOAT_EQ(Ainv.data[2], 1.5f);
    ASSERT_FLOAT_EQ(Ainv.data[3], -0.5f);
}

TEST_P(TestMatrixInverse, CheckLargeInverse) {

    std::vector<float> a(1000000, 0.0f);
    for (int i=0; i<1000; i++) {
        a[i*1000 + ((i + 1) % 1000)] = 1.0f;
    }

    Tensor A = Tensor(a, 1000, 1000, GetParam());

    Tensor Ainv = A.inverse();

    ASSERT_EQ(Ainv.getWidth(), 1000);
    ASSERT_EQ(Ainv.getHeight(), 1000);

    for (int i=0; i<1000; i++) {
        for (int j=0; j<1000; j++) {
            if ((i + 999 - j) % 1000 == 0) {
                ASSERT_FLOAT_EQ(Ainv.data[i*1000 + j], 1.0f);
            } else {
                ASSERT_FLOAT_EQ(Ainv.data[i*1000 + j], 0.0f);
            }
        }
    }
}

TEST_P(TestMatrixInverse, CheckSingularInverse) {

    Tensor A = Tensor({1.0f, 0.0f, 0.0f, 0.0f}, 2, 2, GetParam());

    ASSERT_THROW(A.inverse(), std::runtime_error);
}

TEST_P(TestMatrixInverse, CheckNonSquareInverse) {

    Tensor A = Tensor({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 2, 3, GetParam());

    ASSERT_THROW(A.inverse(), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(TestMatrixInverse, TestMatrixInverse, testing::Values(
    &BasicTensorStrategy::get_instance(),
    &SimdTensorStrategy::get_instance(),
    &ConcurrentRowTensorStrategy::get_instance(),
    &OptimisedTensorStrategy::get_instance(),
    &BlockedSimdTensorStrategy::get_instance()
));
