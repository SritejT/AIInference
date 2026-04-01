#pragma once
#include <vector>
#include "tensor.h"

using namespace std;

class BasicTensor : public Tensor {
public:
    BasicTensor(vector<float> d, size_t h, size_t w);
    BasicTensor(size_t h, size_t w);

    BasicTensor operator+(const BasicTensor& other) const;
    BasicTensor operator*(const BasicTensor& other) const;
};
