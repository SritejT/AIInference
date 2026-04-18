#include "strategies/optimised_tensor_strategy.h"
#define SIMD_THRESHOLD 50
#define CONCURRENCY_THRESHOLD 500000

void OptimisedTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t operations = A->getHeight() * A->getWidth(); 

    if (operations > CONCURRENCY_THRESHOLD) {
        concurrent_strategy.add(A, B, result);
    } else {
        simd_strategy.add(A, B, result);
    }
    
}

void OptimisedTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t operations = A->getHeight() * B->getWidth() * A->getWidth();

    if (operations > CONCURRENCY_THRESHOLD) {
        concurrent_strategy.mult(A, B, result);
    } else {
        simd_strategy.mult(A, B, result);
    }
    
}

void OptimisedTensorStrategy::transpose(const Tensor* A, Tensor* result) const {

    size_t operations = A->getHeight() * A->getWidth();

    if (operations > CONCURRENCY_THRESHOLD) {
        concurrent_strategy.transpose(A, result);
    } else {
        simd_strategy.transpose(A, result);
    }
}

void OptimisedTensorStrategy::apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const {

    size_t operations = A->getHeight() * A->getWidth();

    if (operations > CONCURRENCY_THRESHOLD) {
        concurrent_strategy.apply(f, A, result);
    } else {
        simd_strategy.apply(f, A, result);
    }
}

void OptimisedTensorStrategy::inverse(const Tensor* A, Tensor* result) const {
    simd_strategy.inverse(A, result);
}
