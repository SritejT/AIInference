#include <vector>
#include <iostream>
#include "tensors.h"
#include <algorithm>
#include <arm_neon.h>
using namespace std;

Tensor::Tensor(size_t h, size_t w) {
    width = w;
    height = h;

    data.resize(h*w);
    
}

Tensor::Tensor(vector<float> d, size_t h, size_t w) {
    width = w;
    height = h;
    data = d;
}

size_t Tensor::getWidth() const { return width; }
size_t Tensor::getHeight() const { return height; }


Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result(height, other.getWidth());
    size_t B = 16;


    // Blocked loop with 16^3 = 4096 block size
    for (size_t II = 0; II < height; II += B) {
        for (size_t JJ = 0; JJ < other.getWidth(); JJ += B) {
            for (size_t KK = 0; KK < width; KK += B) {
                
                for (size_t i = II; i < min(II + B, height); i++) {
                    for (size_t k = KK; k < min(KK + B, width); k++) {

                        float32x4_t vx = vdupq_n_f32(data[i*width + k]);

                        size_t j = JJ;
                        size_t simd_limit = min(JJ + B, other.getWidth()) & ~3;

                        for (; j < simd_limit; j+=4) {
                            
                            float32x4_t vy = vld1q_f32(&result.data[i*other.getWidth() + j]);
                            float32x4_t vz = vld1q_f32(&other.data[k*other.getWidth() + j]);

                            vy = vfmaq_f32(vy, vx, vz);

                            vst1q_f32(&result.data[i*other.getWidth() + j], vy);

                        }

                        for (; j < min(JJ + B, other.getWidth()); j++) {
                            result.data[i*other.getWidth() + j] += data[i*width + k] * other.data[k*other.getWidth() + j];
                        }
                        

                    }
                }
            }
        }
    }

    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(height, width);

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            result.data[i*width + j] += other.data[i*width + j];
        }
    }


    return result;
}

vector<float>::const_iterator Tensor::begin() { return data.begin(); }
vector<float>::const_iterator Tensor::end() { return data.end(); }

void Tensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i*width + j] << " ";
        }
        cout << endl;
    }
}

