#include "strategies/basic_simd_tensor_strategy.h"
#include "tensor.h"
#include <arm_neon.h>
#include <cstddef>

void BasicSimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    process_mult_block(A, B, result, 0, 0, height, width);
}

void BasicSimdTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    process_add_block(A, B, result, 0, 0, height, width);
}
