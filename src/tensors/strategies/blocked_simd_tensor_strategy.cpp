#include "strategies/blocked_simd_tensor_strategy.h"

#define BLOCK_SIZE 64 

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
    process_add_block(A, B, result, 0, 0, result->getHeight(), result->getWidth());
}

void BlockedSimdTensorStrategy::process_add_block(
        const Tensor* A, 
        const Tensor* B, 
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t end_row,
        size_t end_col) const {
    simd_strategy.process_add_block(A, B, result, start_row, start_col, end_row, end_col);
}

void BlockedSimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_mult_block(A, B, result, 0, 0, 0, A->getHeight(), B->getWidth(), B->getHeight());
}

void BlockedSimdTensorStrategy::process_mult_block(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t start_k,
        size_t end_row,
        size_t end_col,
        size_t end_k) const {

    size_t result_height = A->getHeight();
    size_t result_width = B->getWidth();

    size_t A_width = A->getWidth(); 


    for (size_t I = start_row; I < end_row; I += BLOCK_SIZE) {
        for (size_t J = start_col; J < end_col; J += BLOCK_SIZE) {
            for (size_t K = start_k; K < end_k; K += BLOCK_SIZE) {
                simd_strategy.process_mult_block(
                        A,
                        B, 
                        result,
                        I, 
                        J, 
                        K, 
                        std::min(I + BLOCK_SIZE, end_row),
                        std::min(J + BLOCK_SIZE, end_col),
                        std::min(K + BLOCK_SIZE, end_k));
            }
        }
    }
}
void BlockedSimdTensorStrategy::transpose(const Tensor* A, Tensor* result) const {
    simd_strategy.process_transpose_block(A, result, 0, A->getHeight());
}

void BlockedSimdTensorStrategy::process_transpose_block(
        const Tensor* A,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {
    simd_strategy.process_transpose_block(A, result, start_row, end_row);
}

void BlockedSimdTensorStrategy::inverse(const Tensor* A, Tensor* result) const {
    simd_strategy.inverse(A, result);
}

void BlockedSimdTensorStrategy::apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const {
    process_apply_block(f, A, result, 0, A->getHeight());
}

void BlockedSimdTensorStrategy::process_apply_block(std::function<float(float)> f, const Tensor* A, Tensor* result, size_t start_row, size_t end_row) const {
    simd_strategy.process_apply_block(f, A, result, start_row, end_row);
}
