#pragma once
#include <vector>
#include "itensor.h"

using namespace std;

class BasicTensor : public ITensor {
public:
    BasicTensor(vector<float> d, size_t h, size_t w);
    BasicTensor(size_t h, size_t w);

    BasicTensor operator+(const BasicTensor& other) const;
    BasicTensor operator*(const BasicTensor& other) const;
};
