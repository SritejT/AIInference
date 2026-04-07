#include <gtest/gtest.h>
#include "parse_safetensors.h"

TEST(ParsingTest, OpenFileTest) {
    SafeTensorsParser parser(std::string(SRC_DIR) + "/models/model.safetensors");
}

TEST(ParsingTest, CheckNumTensorsTest) {
    SafeTensorsParser parser(std::string(SRC_DIR) + "/models/model.safetensors");

    std::vector<Tensor> tensors = parser.parse();

    ASSERT_EQ(tensors.size(), 6);

}

TEST(ParsingTest, CheckParsedValuesTest) {
    SafeTensorsParser parser(std::string(SRC_DIR) + "/models/model.safetensors");
    std::vector<Tensor> tensors = parser.parse();

    const float tol = 1e-7f;

    // input_layer.bias
    auto it = tensors[0].begin();
    EXPECT_NEAR(*it++,  0.007163740694522858f,  tol);
    EXPECT_NEAR(*it++, -0.022049840539693832f,  tol);
    EXPECT_NEAR(*it++,  0.025607815012335777f,  tol);
    EXPECT_NEAR(*it++, -0.013958233408629894f,  tol);

    // input_layer.weight
    it = tensors[1].begin();
    EXPECT_NEAR(*it++,  0.017587387934327126f,  tol);
    EXPECT_NEAR(*it++, -0.016578372567892075f,  tol);
    EXPECT_NEAR(*it++, -0.007738478016108275f,  tol);
    EXPECT_NEAR(*it++,  0.015948316082358360f,  tol);

    // output_layer.bias
    float expected_output_bias[4] = {
        -0.027056507766246796f, -0.033497169613838196f,
         0.028059488162398340f,  0.011518276296555996f,
    };
    it = tensors[4].begin();
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(*it++, expected_output_bias[i], tol);
}
