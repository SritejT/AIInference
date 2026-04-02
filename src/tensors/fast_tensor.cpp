#include <vector>
#include "fast_tensor.h"
#include <arm_neon.h>
#include <thread>
#include "strategies/concurrent_tensor_strategy.h"

using namespace std;

FastTensor::FastTensor(size_t h, size_t w): Tensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    FastTensor result(height, other.getWidth());

    ConcurrentTensorStrategy strategy;
    strategy.mult(this, &other, &result);

    return result;
}

FastTensor FastTensor::operator+(const FastTensor& other) const {

    FastTensor result(this->height, this->width);

    ConcurrentTensorStrategy strategy;
    strategy.add(this, &other, &result);

    return result;
}


