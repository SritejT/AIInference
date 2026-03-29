#include <gtest/gtest.h>
#include "parse_safetensors.h"

using namespace std;

TEST(ParsingTest, OpenFileTest) {
    SafeTensorsParser parser(string(SRC_DIR) + "/models/model.safetensors");
}

TEST(ParsingTest, ParseFileTest) {
    SafeTensorsParser parser(string(SRC_DIR) + "/models/model.safetensors");

    vector<FastTensor> tensors = parser.parse();

    ASSERT_EQ(tensors.size(), 6);

}
