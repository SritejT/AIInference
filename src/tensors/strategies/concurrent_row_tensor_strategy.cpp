#include "strategies/concurrent_row_tensor_strategy.h"
#include <arm_neon.h>
#include <thread>
#include <vector>


void ConcurrentRowTensorStrategy::process_mult_rows(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {

    size_t new_width = B->getWidth();

    size_t A_width = A->getWidth();

    for (size_t i = start_row; i < end_row; i++) {
        size_t j=0;

        for (; j+4 <= new_width; j+=4) {
            
            // Accumulates result[i][j:j+4]
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < A_width; k++) {
                float32x4_t va = vdupq_n_f32(A->data[i * A_width + k]);
                float32x4_t vb = vld1q_f32(&B->data[k * new_width + j]);
                acc = vmlaq_f32(acc, va, vb);
            }

            vst1q_f32(&result->data[i * new_width + j], acc);

        }

        // Do regular matrix mult for all j not covered by 
        // SIMD (i.e. if the j dimension is not a multiple of 4)
        for (; j < new_width; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A_width; k++) {
                sum += A->data[i * A_width + k] * B->data[k * new_width + j];
            }
            result->data[i * new_width + j] = sum;
        }
        
    }
}

void ConcurrentRowTensorStrategy::process_add_rows(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {

    size_t width = A->getWidth();

    size_t i = start_row * width;

    for (; i+4 < end_row * width; i+=4) {
        float32x4_t va = vld1q_f32(&A->data[i]);
        float32x4_t vb = vld1q_f32(&B->data[i]);
        va = vaddq_f32(va, vb);
        vst1q_f32(&result->data[i], va);
    }

    for (; i < end_row * width; i++) {
        result->data[i] = A->data[i] + B->data[i];
    }

}


void ConcurrentRowTensorStrategy::add(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = thread::hardware_concurrency();
    vector<thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(thread(
            &ConcurrentRowTensorStrategy::process_add_rows,
            this,
            A, 
            B,
            result,
            i * height / num_threads,
            (i + 1) * height / num_threads
        ));

    }

    for (auto& t : threads) {
        t.join();
    }
}

void ConcurrentRowTensorStrategy::mult(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = thread::hardware_concurrency();
    vector<thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(thread(
            &ConcurrentRowTensorStrategy::process_mult_rows,
            this,
            A, 
            B,
            result,
            i * height / num_threads,
            (i + 1) * height / num_threads
        ));

    }

    for (auto& t : threads) {
        t.join();
    }
}



