#include <vector>
#include "basic_tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include <stdexcept>
#include <string>
#include <format>

using namespace std;

BasicTensor::BasicTensor(size_t h, size_t w): Tensor(h, w) {}

BasicTensor::BasicTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

BasicTensor BasicTensor::operator*(const BasicTensor& other) const {

    if (width != other.height) {
        string error_msg = format("Matrix multiplication dimensions do not match: [{}, {}] x [{}, {}]", height, width, other.getHeight(), other.getWidth());
        throw runtime_error(error_msg);
    }

    BasicTensor result(height, other.getWidth());

    BasicTensorStrategy strategy;

    strategy.mult(this, &other, &result);

    return result;
}

BasicTensor BasicTensor::operator+(const BasicTensor& other) const {

    if (width != other.width || height != other.height) {
        string error_msg = format("Matrix addition dimensions do not match: [{}, {}] + [{}, {}]", height, width, other.getHeight(), other.getWidth());

        throw runtime_error("Matrix addition dimensions do not match");
    }

    BasicTensor result(height, width);

    BasicTensorStrategy strategy;

    strategy.add(this, &other, &result);

    return result;
}


