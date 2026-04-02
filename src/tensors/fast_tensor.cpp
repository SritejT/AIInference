#include <vector>
#include <memory>
#include "fast_tensor.h"
#include <arm_neon.h>
#include "strategies/concurrent_tensor_strategy.h"
#include "strategies/basic_tensor_strategy.h"

using namespace std;

FastTensor::FastTensor(size_t h, size_t w): Tensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    FastTensor result(height, other.getWidth());

    unique_ptr<TensorStrategy> strategy;
    size_t operations = height * other.getWidth() * width;

    if (operations > 1000000) {
        strategy = make_unique<ConcurrentTensorStrategy>();
    } else {
        strategy = make_unique<BasicTensorStrategy>();
    }

    strategy->mult(this, &other, &result);

    return result;
}

FastTensor FastTensor::operator+(const FastTensor& other) const {

    FastTensor result(this->height, this->width);

    unique_ptr<TensorStrategy> strategy;
    size_t operations = height * other.getWidth();

    if (operations > 1000000) {
        strategy = make_unique<ConcurrentTensorStrategy>();
    } else {
        strategy = make_unique<BasicTensorStrategy>();
    }

    strategy->add(this, &other, &result);

    return result;
}


