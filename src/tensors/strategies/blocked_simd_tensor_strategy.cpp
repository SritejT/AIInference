#include <arm_neon.h>
#include "strategies/blocked_simd_tensor_strategy.h"

#define BLOCK_SIZE 32

void BlockedSimdTensorStrategy::subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const {
    simd_strategy.subtract_rows(A, row1, row2, multiple);
}

void BlockedSimdTensorStrategy::scale_row(Tensor* A, size_t row, float multiple) const {
    simd_strategy.scale_row(A, row, multiple);
}

void BlockedSimdTensorStrategy::swap_rows(Tensor* A, size_t row1, size_t row2) const {
    simd_strategy.swap_rows(A, row1, row2);
}

void BlockedSimdTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {
    simd_strategy.process_add_block(A, B, result, 0, 0, A->getHeight(), A->getWidth());
}

void BlockedSimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {

    size_t result_height = A->getHeight();
    size_t result_width = B->getWidth();

    size_t A_width = A->getWidth(); 


    for (size_t I = 0; I < result_height; I += BLOCK_SIZE) {
        for (size_t J = 0; J < result_width; J += BLOCK_SIZE) {
            for (size_t K = 0; K < A_width; K += BLOCK_SIZE) {
                simd_strategy.process_mult_block(
                        A,
                        B, 
                        result,
                        I, 
                        J, 
                        K, 
                        std::min(I + BLOCK_SIZE, result_height),
                        std::min(J + BLOCK_SIZE, result_width),
                        std::min(K + BLOCK_SIZE, A_width));
            }
        }
    }
 
}

void BlockedSimdTensorStrategy::transpose(const Tensor* A, Tensor* result) const {
    simd_strategy.process_transpose_block(A, result, 0, A->getHeight());
}

void BlockedSimdTensorStrategy::inverse(const Tensor* A, Tensor* result) const {
    simd_strategy.inverse(A, result);
}

void BlockedSimdTensorStrategy::apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const {
    simd_strategy.process_apply_block(f, A, result, 0, A->getHeight());
}


