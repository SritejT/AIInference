#include <strategies/simd_tensor_strategy.h>
#include <arm_sve.h>
#include <arm_neon.h>

void SimdTensorStrategy::process_mult_block(
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
    
    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = start_col; j < end_col; j += svcntw()) {
            svbool_t pg = svwhilelt_b32(j, end_col);
            
            svfloat32_t acc = svld1_f32(pg, &result->data[i * result_width + j]); 

            for (size_t k = start_k; k < end_k; k++) {
                svfloat32_t va = svdup_n_f32(A->data[i * A_width + k]);
                svfloat32_t vb = svld1_f32(pg, &B->data[k * result_width + j]);
                acc = svmla_f32_x(pg, acc, va, vb);
            }

            // Store back
            svst1_f32(pg, &result->data[i * result_width + j], acc);
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

        for (size_t j = 0; j < end_col; j+=svcntw()) {
            svbool_t pg = svwhilelt_b32(j, end_col);
            svfloat32_t va = svld1_f32(pg, &A->data[i * width + j]);
            svfloat32_t vb = svld1_f32(pg, &B->data[i * width + j]);
            va = svadd_f32_x(pg, va, vb);
            svst1_f32(pg, &result->data[i * width + j], va);
        }

    }

}

void SimdTensorStrategy::process_transpose_block(
        const Tensor* A,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {

    size_t i = start_row;
    size_t j = 0;

    size_t width = A->getWidth();
    size_t height = A->getHeight();

    for (; i+4 < end_row; i+=4) {
        for (j=0; j+4 < width; j+=4) {

            // load 4 rows
            float32x4_t r0 = vld1q_f32(&A->data[i * width + j]);
            float32x4_t r1 = vld1q_f32(&A->data[(i + 1) * width + j]);
            float32x4_t r2 = vld1q_f32(&A->data[(i + 2) * width + j]);
            float32x4_t r3 = vld1q_f32(&A->data[(i + 3) * width + j]);

            // pairwise transpose
            float32x4x2_t t01 = vtrnq_f32(r0, r1);  // {a0 b0 a2 b2}, {a1 b1 a3 b3}
            float32x4x2_t t23 = vtrnq_f32(r2, r3);  // {c0 d0 c2 d2}, {c1 d1 c3 d3}

            // combine low/high halves
            float32x4_t o0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t o1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t o2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t o3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));

            // store
            vst1q_f32(&result->data[j * height + i], o0);
            vst1q_f32(&result->data[(j + 1) * height + i], o1);
            vst1q_f32(&result->data[(j + 2) * height + i], o2);
            vst1q_f32(&result->data[(j + 3) * height + i], o3);

        }

        for (; j < width; j++) {
            result->data[j * height + i] = A->data[i * width + j];
            result->data[j * height + i + 1] = A->data[(i + 1) * width + j];
            result->data[j * height + i + 2] = A->data[(i + 2) * width + j];
            result->data[j * height + i + 3] = A->data[(i + 3) * width + j];
        }
    }

    for (; i < end_row; i++) {
        for (j = 0; j < width; j++) {
            result->data[j * height + i] = A->data[i * width + j];
        }
    }
}

void SimdTensorStrategy::process_apply_block(
        std::function<float(float)> f,
        const Tensor* A,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {

    size_t width = A->getWidth();


    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < width; j++) {
            result->data[i * width + j] = f(A->data[i * width + j]);
        }
    }

}

void SimdTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_add_block(A, B, result, 0, 0, A->getHeight(), A->getWidth());
}

void SimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_mult_block(A, B, result, 0, 0, 0, A->getHeight(), B->getWidth(), A->getWidth());
}

void SimdTensorStrategy::transpose(const Tensor* A, Tensor* result) const {
    process_transpose_block(A, result, 0, A->getHeight());
}

void SimdTensorStrategy::subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const {

    size_t width = A->getWidth();
    size_t i = 0;

    for (; i+4 <= width; i+=4) {

        float32x4_t vmul = vdupq_n_f32(-multiple);
        float32x4_t r1 = vld1q_f32(&A->data[row1 * width + i]);
        float32x4_t r2 = vld1q_f32(&A->data[row2 * width + i]);

        r1 = vmlaq_f32(r1, r2, vmul);
        vst1q_f32(&A->data[row1 * width + i], r1);
    }

    for (; i < width; i++) {
        A->data[row1 * width + i] -= multiple * A->data[row2 * width + i];
    }
}

void SimdTensorStrategy::scale_row(Tensor* A, size_t row, float multiple) const {

    size_t width = A->getWidth();
    size_t i = 0;

    for (; i+4 <= width; i+=4) {

        float32x4_t vmul = vdupq_n_f32(multiple);
        float32x4_t r1 = vld1q_f32(&A->data[row * width + i]);

        r1 = vmulq_f32(r1, vmul);

        vst1q_f32(&A->data[row * width + i], r1);
    }

    for (; i < width; i++) {
        A->data[row * width + i] *= multiple;
    }
}

void SimdTensorStrategy::swap_rows(Tensor* A, size_t row1, size_t row2) const {

    size_t width = A->getWidth();
    size_t i = 0;

    for (; i+4 <= width; i+=4) {
        float32x4_t r1 = vld1q_f32(&A->data[row1 * width + i]);
        float32x4_t r2 = vld1q_f32(&A->data[row2 * width + i]);

        vst1q_f32(&A->data[row1 * width + i], r2);
        vst1q_f32(&A->data[row2 * width + i], r1);
    }

    for (; i < width; i++) {
        std::swap(A->data[row1 * width + i], A->data[row2 * width + i]);
    }
}


void SimdTensorStrategy::apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const {
    process_apply_block(f, A, result, 0, A->getHeight());
}
