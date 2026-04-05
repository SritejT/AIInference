#include "strategies/optimised_tensor_strategy.h"

void OptimisedTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t operations = A->getHeight() * A->getWidth(); 

    if (operations > 1000000) {
        concurrent_strategy.add(A, B, result);
    } else {
        simd_strategy.add(A, B, result);
    }
    
}

void OptimisedTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t operations = A->getHeight() * B->getWidth() * A->getWidth();

    if (operations > 1000000) {
        concurrent_strategy.mult(A, B, result);
    } else {
        simd_strategy.mult(A, B, result);
    }
    
}
