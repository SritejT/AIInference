#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

#include <gtest/gtest.h>

class MatrixMultTest : public testing::TestWithParam<std::shared_ptr<TensorStrategy>> {};

TEST_P(MatrixMultTest, SmallSquareMatrixMultTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = a * b;
    ASSERT_EQ(result.getWidth(), 2);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}

TEST_P(MatrixMultTest, SmallRectMatrixMultTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 2, 4, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = a * b;
    ASSERT_EQ(result.getWidth(), 4);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[8] = {23.0f, 26.0f, 29.0f, 32.0f, 51.0f, 58.0f, 65.0f, 72.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}

TEST_P(MatrixMultTest, LargeSquareMatrixMultTest) {
    std::vector<float> a = std::vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i+=1001) {
        a[i] = 1.0f;
    }

    std::vector<float> b = std::vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i++) {
        b[i] = static_cast<float>(i);
    }

    Tensor A = Tensor(a, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A * B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, static_cast<float>(j));
        j++;
    }
}

TEST_P(MatrixMultTest, LargeRectMatrixMultTest) {
    std::vector<float> a = std::vector<float>(1000000, 1.0f);
    std::vector<float> b = std::vector<float>(1000000, 1.0f);

    Tensor A = Tensor(a, 1, 1000000, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000000, 1, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A * B;
    ASSERT_EQ(result.getWidth(), 1);
    ASSERT_EQ(result.getHeight(), 1);

    EXPECT_EQ(*result.begin(), 1000000.0f);
    
}


// Test whether loop blocking and SIMD vectorization works with matrices
// whose numbers of elements are not divisible by the block size or 
// number of SIMD registers
TEST_P(MatrixMultTest, PrimeSizeMatrixMultTest) {
    std::vector<float> a = std::vector<float>(67, 1.0f);
    std::vector<float> b = std::vector<float>(67, 1.0f);

    Tensor A = Tensor(a, 67, 1, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1, 67, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A * B;
    ASSERT_EQ(result.getWidth(), 67);
    ASSERT_EQ(result.getHeight(), 67);

    for (auto r : result) {
        EXPECT_EQ(r, 1.0f);
    }
}

TEST_P(MatrixMultTest, NegativeValuesMatrixMultTest) {
    std::vector<float> a = std::vector<float>(1000000, -3.0f);
    std::vector<float> b = std::vector<float>(1000000, -67.0f);

    Tensor A = Tensor(a, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000, 1000, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A * B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    for (auto r : result) {
        EXPECT_EQ(r, 201000.0f);
    }
}

TEST_P(MatrixMultTest, LargeValuesMatrixMultTest) {
    std::vector<float> a = std::vector<float>(100, 20000.0f);
    std::vector<float> b = std::vector<float>(100, 10000.0f);

    Tensor A = Tensor(a, 10, 10, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 10, 10, std::shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A * B;
    ASSERT_EQ(result.getWidth(), 10);
    ASSERT_EQ(result.getHeight(), 10);

    for (auto r : result) {
        EXPECT_EQ(r, 2e9f);
    }
}

TEST_P(MatrixMultTest, InvalidMultTest) {
    std::vector<float> a = std::vector<float>(1000000, 1.0f);
    std::vector<float> b = std::vector<float>(1000000, 1.0f);

    Tensor A = Tensor(a, 1, 1000000, std::shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1, 1000000, std::shared_ptr<TensorStrategy>(GetParam()));

    ASSERT_THROW(A * B, std::runtime_error);
}

INSTANTIATE_TEST_CASE_P(TestAllMultStrategies, MatrixMultTest, testing::Values(
    std::make_shared<BasicTensorStrategy>(),
    std::make_shared<SimdTensorStrategy>(),
    std::make_shared<ConcurrentRowTensorStrategy>(),
    std::make_shared<OptimisedTensorStrategy>(),
    std::make_shared<BlockedSimdTensorStrategy>()));


