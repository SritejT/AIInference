#include "fast_tensor.h"
#include <gtest/gtest.h>

using namespace std;

TEST(MatrixMultTest, SmallSquareMatrixMultTest) {
    FastTensor a = FastTensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    FastTensor b = FastTensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);

    FastTensor result = a * b;
    ASSERT_EQ(result.getWidth(), 2);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}

TEST(MatrixMultTest, SmallRectMatrixMultTest) {
    FastTensor a = FastTensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    FastTensor b = FastTensor({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 2, 4);

    FastTensor result = a * b;
    ASSERT_EQ(result.getWidth(), 4);
    ASSERT_EQ(result.getHeight(), 2);

    float expected[8] = {23.0f, 26.0f, 29.0f, 32.0f, 51.0f, 58.0f, 65.0f, 72.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}

TEST(MatrixMultTest, LargeSquareMatrixMultTest) {
    vector<float> a = vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i+=1001) {
        a[i] = 1.0f;
    }

    vector<float> b = vector<float>(1000000, 0.0f);
    for (unsigned long i=0; i<1000000; i++) {
        b[i] = static_cast<float>(i);
    }

    FastTensor A = FastTensor(a, 1000, 1000);
    FastTensor B = FastTensor(b, 1000, 1000);

    FastTensor result = A * B;
    ASSERT_EQ(result.getWidth(), 1000);
    ASSERT_EQ(result.getHeight(), 1000);

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, static_cast<float>(j));
        j++;
    }
}

TEST(MatrixMultTest, LargeRectMatrixMultTest) {
    vector<float> a = vector<float>(1000000, 1.0f);
    vector<float> b = vector<float>(1000000, 1.0f);

    FastTensor A = FastTensor(a, 1, 1000000);
    FastTensor B = FastTensor(b, 1000000, 1);

    FastTensor result = A * B;
    ASSERT_EQ(result.getWidth(), 1);
    ASSERT_EQ(result.getHeight(), 1);

    EXPECT_EQ(*result.begin(), 1000000.0f);
    
}
