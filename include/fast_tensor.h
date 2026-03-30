#pragma once
#include "itensor.h"
#include <queue>

using namespace std;

class FastTensor : public ITensor {
private:
    void process_rows(FastTensor* result, const FastTensor* other, size_t start_row, size_t end_row) const;
public:
    FastTensor(size_t h, size_t w);
    FastTensor(vector<float> d, size_t h, size_t w);

    FastTensor operator+(const FastTensor& other) const;
    FastTensor operator*(const FastTensor& other) const;
};
