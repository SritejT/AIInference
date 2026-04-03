#include "strategies/concurrent_blocked_tensor_strategy.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

void ConcurrentBlockedTensorStrategy::process_mult_block(
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

void ConcurrentBlockedTensorStrategy::process_add_block(
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

void ConcurrentBlockedTensorStrategy::add(
        const Tensor *A,
        const Tensor *B,
        Tensor *result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    size_t num_cores = thread::hardware_concurrency();

    size_t grid_dim = sqrt(num_cores);

    vector<thread> threads;

    for (int i=0; i < grid_dim * grid_dim; i++) {
        size_t grid_i = i / grid_dim;
        size_t grid_j = i % grid_dim;

        size_t start_row = grid_i * height / grid_dim;
        size_t end_row = (grid_i + 1) * height / grid_dim;
        size_t start_col = grid_j * width / grid_dim;
        size_t end_col = (grid_j + 1) * width / grid_dim;

        threads.push_back(thread(
            &ConcurrentBlockedTensorStrategy::process_add_block,
            this,
            A,
            B,
            result,
            start_row,
            start_col,
            end_row,
            end_col
        ));
    }        

    for (auto& t : threads) {
        t.join();
    }
}

void ConcurrentBlockedTensorStrategy::mult(
        const Tensor *A,
        const Tensor *B,
        Tensor *result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    size_t num_cores = thread::hardware_concurrency();

    size_t grid_dim = sqrt(num_cores);

    vector<thread> threads;

    for (int i=0; i < grid_dim * grid_dim; i++) {
        size_t grid_i = i / grid_dim;
        size_t grid_j = i % grid_dim;

        size_t start_row = grid_i * height / grid_dim;
        size_t end_row = (grid_i + 1) * height / grid_dim;
        size_t start_col = (grid_j * width / grid_dim) / 4 * 4;

        size_t end_col; 
        if (grid_j + 1 == grid_dim) {
            end_col = width;
        } else {
            end_col = (grid_j + 1) * width / grid_dim;
        }

        threads.push_back(thread(
            &ConcurrentBlockedTensorStrategy::process_mult_block,
            this,
            A,
            B,
            result,
            start_row,
            start_col,
            end_row,
            end_col
        ));
    }        

    for (auto& t : threads) {
        t.join();
    }
}
