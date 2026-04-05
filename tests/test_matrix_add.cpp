#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include "strategies/basic_simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/concurrent_blocked_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"

#include <gtest/gtest.h>

using namespace std;

class MatrixAddTest : public testing::TestWithParam<shared_ptr<TensorStrategy>> {};

TEST_P(MatrixAddTest, SmallSquareMatrixAddTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2, shared_ptr<TensorStrategy>(GetParam()));
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = a + b;
    ASSERT_EQ(result.getWidth(), 2);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[4] = {6.0f, 8.0f, 10.0f, 12.0f};

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }
}

TEST_P(MatrixAddTest, SmallRectMatrixAddTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3, shared_ptr<TensorStrategy>(GetParam()));
    Tensor b = Tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 2, 3, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = a + b;
    ASSERT_EQ(result.getWidth(), 3);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[6] = {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f};

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }
}

TEST_P(MatrixAddTest, LargeSquareMatrixAddTest) {
    vector<float> a(1000000);
    vector<float> b(1000000);
    for (unsigned long i = 0; i < 1000000; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    Tensor A = Tensor(a, 1000, 1000, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000, 1000, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, static_cast<float>(j) * 2.0f);
        j++;
    }
}

TEST_P(MatrixAddTest, LargeRectMatrixAddTest) {
    vector<float> a(1000000, 1.0f);
    vector<float> b(1000000, 1.0f);

    Tensor A = Tensor(a, 1, 1000000, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1, 1000000, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1000000);
    ASSERT_EQ(result.getHeight(), 1);

    for (auto r : result) {
        EXPECT_EQ(r, 2.0f);
    }
}

// Test whether addition works with matrices whose number of elements
// is not divisible by the block size or number of SIMD registers
TEST_P(MatrixAddTest, PrimeSizeMatrixAddTest) {
    vector<float> a(67, 1.0f);
    vector<float> b(67, 2.0f);

    Tensor A = Tensor(a, 67, 1, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 67, 1, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1);
    ASSERT_EQ(result.getHeight(), 67);

    for (auto r : result) {
        EXPECT_EQ(r, 3.0f);
    }
}

TEST_P(MatrixAddTest, NegativeValuesMatrixAddTest) {
    vector<float> a = vector<float>(1000000, -3.0f);
    vector<float> b = vector<float>(1000000, -67.0f);

    Tensor A = Tensor(a, 1000, 1000, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 1000, 1000, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    for (auto r : result) {
        EXPECT_EQ(r, -70.0f);
    }
}

TEST_P(MatrixAddTest, LargeValuesMatrixAddTest) {
    vector<float> a = vector<float>(100, 20000.0f);
    vector<float> b = vector<float>(100, 10000.0f);

    Tensor A = Tensor(a, 10, 10, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 10, 10, shared_ptr<TensorStrategy>(GetParam()));

    Tensor result = A + B;
    ASSERT_EQ(result.getWidth(), 10);
    ASSERT_EQ(result.getHeight(), 10);

    for (auto r : result) {
        EXPECT_EQ(r, 30000.0f);
    }
}

TEST_P(MatrixAddTest, InvalidAddTest) {
    vector<float> a = vector<float>(100, 20000.0f);
    vector<float> b = vector<float>(100, 10000.0f);

    Tensor A = Tensor(a, 10, 10, shared_ptr<TensorStrategy>(GetParam()));
    Tensor B = Tensor(b, 100, 1, shared_ptr<TensorStrategy>(GetParam()));

    ASSERT_THROW(A + B, std::runtime_error);
}

INSTANTIATE_TEST_CASE_P(TestAllAddStrategies, MatrixAddTest, testing::Values(
            make_shared<BasicTensorStrategy>(),
            make_shared<BasicSimdTensorStrategy>(),
            make_shared<ConcurrentRowTensorStrategy>(),
            make_shared<ConcurrentBlockedTensorStrategy>(),
            make_shared<OptimisedTensorStrategy>()));


