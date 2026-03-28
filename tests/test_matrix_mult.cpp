#include "tensors.h"

#include <gtest/gtest.h>

using namespace std;

TEST(MatrixMultTest, SmallSquareMatrixMultTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);

    Tensor result = a * b;
    EXPECT_EQ(result.getWidth(), 2);
    EXPECT_EQ(result.getHeight(), 2);

    float expected[4] = {19.0f, 22.0f, 43.0f, 50.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}

TEST(MatrixMultTest, SmallRectMatrixMultTest) {
    Tensor a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    Tensor b = Tensor({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, 2, 4);

    Tensor result = a * b;
    EXPECT_EQ(result.getWidth(), 4);
    EXPECT_EQ(result.getHeight(), 2);

    float expected[8] = {23.0f, 26.0f, 29.0f, 32.0f, 51.0f, 58.0f, 65.0f, 72.0f};

    int j=0;
    for (auto r : result) {
        EXPECT_EQ(r, expected[j]);
        j++;
    }

}
