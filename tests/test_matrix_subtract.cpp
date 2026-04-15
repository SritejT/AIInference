#include <gtest/gtest.h>
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

class MatrixSubtractTest : public testing::TestWithParam<std::shared_ptr<TensorStrategy>> {};

TEST_P(MatrixSubtractTest, SmallMatrices) {

    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, GetParam());
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2, GetParam());

    Tensor result = a - b;

    ASSERT_EQ(result.getWidth(), 2);
    ASSERT_EQ(result.getHeight(), 2);
    ASSERT_EQ(result.data[0], -4.0f);
    ASSERT_EQ(result.data[1], -4.0f);
    ASSERT_EQ(result.data[2], -4.0f);
    ASSERT_EQ(result.data[3], -4.0f);
}

TEST_P(MatrixSubtractTest, LargeSquareMatrices) {

    std::vector<float> a = std::vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i++) {
        a[i] = i;
    }

    std::vector<float> b = std::vector<float>(1000000, 1.0f);

    Tensor A = Tensor(a, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A - B;
    
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    for (int i=0; i<1000000; i++) {
        ASSERT_EQ(result.data[i], static_cast<float>(i-1));
    }

}

TEST_P(MatrixSubtractTest, LargeNonSquareMatrices) {

    std::vector<float> a = std::vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i++) {
        a[i] = i;
    }

    std::vector<float> b = std::vector<float>(1000000, 1.0f);

    Tensor A = Tensor(a, 1000000, 1, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000000, 1, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A - B;
    
    ASSERT_EQ(result.getHeight(), 1000000);
    ASSERT_EQ(result.getWidth(), 1);

    for (int i=0; i<1000000; i++) {
        ASSERT_EQ(result.data[i], static_cast<float>(i-1));
    }
}

TEST_P(MatrixSubtractTest, PrimeSizeMatrices) {

    int n = 67;

    std::vector<float> a = std::vector<float>(n * n, 0.0f);
    for (unsigned long i=0; i<n*n; i++) {
        a[i] = i;
    }

    std::vector<float> b = std::vector<float>(n * n, 1.0f);

    Tensor A = Tensor(a, n, n, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, n, n, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A - B;
    
    ASSERT_EQ(result.getWidth(), n);
    ASSERT_EQ(result.getHeight(), n);

    for (int i=0; i<n*n; i++) {
        ASSERT_EQ(result.data[i], static_cast<float>(i-1));
    }
}

INSTANTIATE_TEST_SUITE_P(BasicTensorStrategy, MatrixSubtractTest, testing::Values(
            std::make_shared<BasicTensorStrategy>(),
            std::make_shared<SimdTensorStrategy>(),
            std::make_shared<ConcurrentRowTensorStrategy>(),
            std::make_shared<OptimisedTensorStrategy>(),
            std::make_shared<BlockedSimdTensorStrategy>()));



