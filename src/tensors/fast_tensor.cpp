#include <vector>
#include "fast_tensor.h"
#include <arm_neon.h>
#include <thread>
using namespace std;

FastTensor::FastTensor(size_t h, size_t w): ITensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): ITensor(d, h, w) {}

void FastTensor::process_rows(FastTensor* result, const FastTensor* other, size_t start_row, size_t end_row) const {

    for (size_t i = start_row; i < end_row; i++) {
        size_t j=0;

        for (; j+4 <= other->getWidth(); j+=4) {
            
            // Accumulates result[i][j:j+4]
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (size_t k = 0; k < width; k++) {
                float32x4_t va = vdupq_n_f32(data[i * width + k]);
                float32x4_t vb = vld1q_f32(&other->data[k * other->getWidth() + j]);
                acc = vmlaq_f32(acc, va, vb);
            }

            vst1q_f32(&result->data[i * other->getWidth() + j], acc);

        }

        // Do regular matrix mult for all j not covered by SIMD (i.e. if the j dimension is 
        // not a multiple of 4)
        for (; j < other->getWidth(); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < width; k++) {
                sum += data[i * width + k] * other->data[k * other->getWidth() + j];
            }
            result->data[i * other->getWidth() + j] = sum;
        }
        
    }
}

FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    shared_ptr<FastTensor> result(new FastTensor(height, other.getWidth()));

    size_t num_threads = thread::hardware_concurrency();

    vector<thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(thread(
            &FastTensor::process_rows,
            this,
            result.get(), 
            &other,
            i * height / num_threads,
            (i + 1) * height / num_threads
        ));

    }

    for (auto& t : threads) {
        t.join();
    }

    FastTensor result_copy = *result;

    return result_copy;
}

FastTensor FastTensor::operator+(const FastTensor& other) const {

    FastTensor result = *this;

    size_t simd_limit = (width * height) & ~3;
    for (size_t i = 0; i < simd_limit; i+=4) {
        float32x4_t va = vld1q_f32(&result.data[i]);
        float32x4_t vb = vld1q_f32(&other.data[i]);
        va = vaddq_f32(va, vb);
        vst1q_f32(&result.data[i], va);
    }

    for (size_t i = simd_limit; i < width * height; i++) {
        result.data[i] += other.data[i];
    }

    return result;
}


