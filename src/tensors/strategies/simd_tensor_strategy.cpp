#include <strategies/simd_tensor_strategy.h>
#include <arm_neon.h>

void SimdTensorStrategy::process_mult_block(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t end_row,
        size_t end_col) const {

    size_t result_height = A->getHeight();
    size_t result_width = B->getWidth();

    size_t A_width = A->getWidth(); 
    
    for (int i = start_row; i < end_row; i++) {

        size_t j = start_col;

        for (; j+4 < end_col; j+=4) {

            // Accumulates result[i][j:j+4]
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < A_width; k++) {
                float32x4_t va = vdupq_n_f32(A->data[i * A_width + k]);
                float32x4_t vb = vld1q_f32(&B->data[k * result_width + j]);
                acc = vmlaq_f32(acc, va, vb);
            }

            vst1q_f32(&result->data[i * result_width + j], acc);
        }

        for (; j < end_col; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A_width; k++) {
                sum += A->data[i * A_width + k] * B->data[k * result_width + j];
            }
            result->data[i * result_width + j] = sum;
        }
    }
}

void SimdTensorStrategy::process_add_block(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t end_row,
        size_t end_col) const {

    size_t width = A->getWidth();
    
    for (size_t i = start_row; i < end_row; i++) {

        size_t j = start_col;

        for (; j+4 < end_col; j+=4) {
            float32x4_t va = vld1q_f32(&A->data[i * width + j]);
            float32x4_t vb = vld1q_f32(&B->data[i * width + j]);
            va = vaddq_f32(va, vb);
            vst1q_f32(&result->data[i * width + j], va);
        }

        for (; j < end_col; j++) {
            result->data[i * width + j] = A->data[i * width + j] + B->data[i * width + j];
        }
    }

}

void SimdTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_add_block(A, B, result, 0, 0, A->getHeight(), A->getWidth());
}

void SimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_mult_block(A, B, result, 0, 0, A->getHeight(), B->getWidth());
}
