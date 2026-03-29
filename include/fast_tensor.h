#pragma once
#include "itensor.h"

using namespace std;

class FastTensor : public ITensor {
public:
    FastTensor(size_t h, size_t w);
    FastTensor(vector<float> d, size_t h, size_t w);

    FastTensor operator+(const FastTensor& other) const;
    FastTensor operator*(const FastTensor& other) const;
};
