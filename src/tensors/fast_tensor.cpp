#include <vector>
#include "fast_tensor.h"
#include <arm_neon.h>
#include <thread>
using namespace std;

FastTensor::FastTensor(size_t h, size_t w): ITensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): ITensor(d, h, w) {}

void FastTensor::process_work_queue(FastTensor* result, const FastTensor* other, queue<pair<size_t, size_t>>* work_queue) const {

    while (!work_queue->empty()) {

        pair<size_t, size_t> next_pair = work_queue->front();
        work_queue->pop();

        size_t i = next_pair.first;
        size_t j = next_pair.second;

        // Accumulates result[i][j:j+4]
        float32x4_t acc = vdupq_n_f32(0.0f);

        for (size_t k = 0; k < width; k++) {
            float32x4_t va = vdupq_n_f32(data[i * width + k]);
            float32x4_t vb = vld1q_f32(&other->data[k * other->getWidth() + j]);
            acc = vmlaq_f32(acc, va, vb);
        }

        vst1q_f32(&result->data[i * other->getWidth() + j], acc);
        
    }
}

FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    shared_ptr<FastTensor> result(new FastTensor(height, other.getWidth()));

    size_t num_threads = thread::hardware_concurrency();
    shared_ptr<vector<queue<pair<size_t, size_t>>>> work_queues(new vector<queue<pair<size_t, size_t>>>(num_threads));

    for (size_t i = 0; i < height; i++) {

        size_t j = 0;
        size_t counter = 0;
        
        for (; j+4 < other.getWidth(); j+=4) {

            (*work_queues)[counter].push(pair<size_t, size_t>(i, j));
            counter = (counter + 1) % num_threads;
        }

        // Do regular matrix mult for all j not covered by SIMD (i.e. if the j dimension is 
        // not a multiple of 4)
        for (; j < other.getWidth(); j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < width; k++) {
                sum += data[i * width + k] * other.data[k * other.getWidth() + j];
            }
            result->data[i * other.getWidth() + j] = sum;
        }
        
    }

    vector<thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        queue<pair<size_t, size_t>>* wq = &(*work_queues)[i];

        threads.push_back(thread(
            &FastTensor::process_work_queue,
            this,
            result.get(), 
            &other, 
            wq
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


