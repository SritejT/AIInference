#include <vector>
#include "fast_tensor.h"
#include <arm_neon.h>
using namespace std;

FastTensor::FastTensor(size_t h, size_t w): ITensor(h, w) {}

FastTensor::FastTensor(vector<float> d, size_t h, size_t w): ITensor(d, h, w) {}

FastTensor FastTensor::operator*(const FastTensor& other) const {
    
    FastTensor result(height, other.getWidth());

    for (size_t i = 0; i < height; i++) {

        size_t simd_limit = other.getWidth() & ~3;

        vector<float32x4_t> accs(simd_limit / 4, vdupq_n_f32(0.0f));

        for (size_t k = 0; k < width; k++) {
            
            float x = data[i * width + k];
            float32x4_t vx = vdupq_n_f32(x);

            size_t j = 0;

            for (; j < simd_limit; j+=4) {

                float32x4_t vy = vld1q_f32(&other.data[k*other.getWidth() + j]);

                accs[j/4] = vfmaq_f32(accs[j/4], vx, vy);
            }

            for (; j < other.getWidth(); j++) {
                result.data[i*other.getWidth() + j] += x * other.data[k*other.getWidth() + j];
            }
        }

        for (size_t j = 0; j < simd_limit; j += 4) {
            vst1q_f32(&result.data[i*other.getWidth() + j], accs[j/4]);
        }
    }

    return result;
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


