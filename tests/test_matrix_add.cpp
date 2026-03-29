#include "fast_tensor.h"
#include <gtest/gtest.h>

using namespace std;

TEST(MatrixAddTest, SmallSquareMatrixAddTest) {
    FastTensor a = FastTensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    FastTensor b = FastTensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);

    FastTensor result = a + b;
    ASSERT_EQ(result.getWidth(), 2);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[4] = {6.0f, 8.0f, 10.0f, 12.0f};

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }
}

TEST(MatrixAddTest, SmallRectMatrixAddTest) {
    FastTensor a = FastTensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
    FastTensor b = FastTensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 2, 3);

    FastTensor result = a + b;
    ASSERT_EQ(result.getWidth(), 3);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[6] = {8.0f, 10.0f, 12.0f, 14.0f, 16.0f, 18.0f};

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }
}

TEST(MatrixAddTest, LargeSquareMatrixAddTest) {
    vector<float> a(1000000);
    vector<float> b(1000000);
    for (unsigned long i = 0; i < 1000000; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    FastTensor A = FastTensor(a, 1000, 1000);
    FastTensor B = FastTensor(b, 1000, 1000);

    FastTensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    int j = 0;
    for (auto r : result) {
        EXPECT_EQ(r, static_cast<float>(j) * 2.0f);
        j++;
    }
}

TEST(MatrixAddTest, LargeRectMatrixAddTest) {
    vector<float> a(1000000, 1.0f);
    vector<float> b(1000000, 1.0f);

    FastTensor A = FastTensor(a, 1, 1000000);
    FastTensor B = FastTensor(b, 1, 1000000);

    FastTensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1000000);
    ASSERT_EQ(result.getHeight(), 1);

    for (auto r : result) {
        EXPECT_EQ(r, 2.0f);
    }
}

// Test whether addition works with matrices whose number of elements
// is not divisible by the block size or number of SIMD registers
TEST(MatrixAddTest, PrimeSizeMatrixAddTest) {
    vector<float> a(67, 1.0f);
    vector<float> b(67, 2.0f);

    FastTensor A = FastTensor(a, 67, 1);
    FastTensor B = FastTensor(b, 67, 1);

    FastTensor result = A + B;
    ASSERT_EQ(result.getWidth(), 1);
    ASSERT_EQ(result.getHeight(), 67);

    for (auto r : result) {
        EXPECT_EQ(r, 3.0f);
    }
}
