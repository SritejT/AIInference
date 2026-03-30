#pragma once
#include "itensor.h"
#include <queue>

using namespace std;

class FastTensor : public ITensor {
private:
    void process_work_queue(FastTensor* result, const FastTensor* other, queue<pair<size_t, size_t>>* work_queue) const;
public:
    FastTensor(size_t h, size_t w);
    FastTensor(vector<float> d, size_t h, size_t w);

    FastTensor operator+(const FastTensor& other) const;
    FastTensor operator*(const FastTensor& other) const;
};
