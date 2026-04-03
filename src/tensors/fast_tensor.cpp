#include <vector>
#include <memory>
#include "fast_tensor.h"
#include <arm_neon.h>
#include <format>
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/concurrent_blocked_tensor_strategy.h"
#include "strategies/basic_tensor_strategy.h"

using namespace std;

FastTensor::FastTensor(size_t h, size_t w): Tensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

FastTensor FastTensor::operator*(const FastTensor& other) const {

    if (width != other.height) {
        string error_msg = format("Matrix multiplication dimensions do not match: [{}, {}] x [{}, {}]", height, width, other.getHeight(), other.getWidth());
        throw runtime_error(error_msg);
    }   

    FastTensor result(height, other.getWidth());

    unique_ptr<TensorStrategy> strategy;
    size_t operations = height * other.getWidth() * width;

    if (operations > 1000000) {
        strategy = make_unique<ConcurrentRowTensorStrategy>();
    } else {
        strategy = make_unique<BasicTensorStrategy>();
    }

    strategy->mult(this, &other, &result);

    return result;
}

FastTensor FastTensor::operator+(const FastTensor& other) const {

    if (width != other.width || height != other.height) {
        string error_msg = format("Matrix addition dimensions do not match: [{}, {}] + [{}, {}]", height, width, other.getHeight(), other.getWidth());

        throw runtime_error("Matrix addition dimensions do not match");
    }

    FastTensor result(this->height, this->width);

    unique_ptr<TensorStrategy> strategy;
    size_t operations = height * other.getWidth();

    if (operations > 1000000) {
        strategy = make_unique<ConcurrentRowTensorStrategy>();
    } else {
        strategy = make_unique<BasicTensorStrategy>();
    }

    strategy->add(this, &other, &result);

    return result;
}


