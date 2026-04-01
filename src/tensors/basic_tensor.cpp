#include <vector>
#include "basic_tensor.h"
#include "strategies/basic_tensor_strategy.h"

using namespace std;

BasicTensor::BasicTensor(size_t h, size_t w): Tensor(h, w) {}

BasicTensor::BasicTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

BasicTensor BasicTensor::operator*(const BasicTensor& other) const {
    BasicTensor result(height, other.getWidth());

    BasicTensorStrategy strategy;

    strategy.mult(this, &other, &result);

    return result;
}

BasicTensor BasicTensor::operator+(const BasicTensor& other) const {
    BasicTensor result(height, width);

    BasicTensorStrategy strategy;

    strategy.add(this, &other, &result);

    return result;
}


