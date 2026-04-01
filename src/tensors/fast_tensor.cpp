#include <vector>
#include "fast_tensor.h"
#include <arm_neon.h>
#include <thread>

using namespace std;

FastTensor::FastTensor(size_t h, size_t w): Tensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): Tensor(d, h, w) {}

void FastTensor::process_mult_rows(FastTensor* result, const FastTensor* other, size_t start_row, size_t end_row) const {

    size_t new_width = other->getWidth();
    for (size_t i = start_row; i < end_row; i++) {
        size_t j=0;

        for (; j+4 <= new_width; j+=4) {
            
            // Accumulates result[i][j:j+4]
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < width; k++) {
                float32x4_t va = vdupq_n_f32(data[i * width + k]);
                float32x4_t vb = vld1q_f32(&other->data[k * new_width + j]);
                acc = vmlaq_f32(acc, va, vb);
            }

            vst1q_f32(&result->data[i * new_width + j], acc);

        }

        // Do regular matrix mult for all j not covered by SIMD (i.e. if the j dimension is 
        // not a multiple of 4)
        for (; j < new_width; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < width; k++) {
                sum += data[i * width + k] * other->data[k * new_width + j];
            }
            result->data[i * new_width + j] = sum;
        }
        
    }
}

void FastTensor::process_add_rows(FastTensor* result, const FastTensor* other, size_t start_row, size_t end_row) const {

    size_t i = start_row * width;

    for (; i+4 < end_row * width; i+=4) {
        float32x4_t va = vld1q_f32(&result->data[i]);
        float32x4_t vb = vld1q_f32(&other->data[i]);
        va = vaddq_f32(va, vb);
        vst1q_f32(&result->data[i], va);
    }

    for (; i < end_row * width; i++) {
        result->data[i] += other->data[i];
    }

}



FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    FastTensor result(height, other.getWidth());

    size_t num_threads = thread::hardware_concurrency();

    vector<thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(thread(
            &FastTensor::process_mult_rows,
            this,
            &result, 
            &other,
            i * height / num_threads,
            (i + 1) * height / num_threads
        ));

    }

    for (auto& t : threads) {
        t.join();
    }

    return result;
}

FastTensor FastTensor::operator+(const FastTensor& other) const {

    size_t num_threads = thread::hardware_concurrency();

    vector<thread> threads;

    FastTensor result = *this;

    for (size_t i = 0; i < num_threads; i++) {
        threads.push_back(thread(
            &FastTensor::process_add_rows,
            this,
            &result,
            &other,
            i * height / num_threads,
            (i + 1) * height / num_threads
            
        ));
    }

    for (auto& t : threads) {
        t.join();
    }

    return result;
}


