#include <vector>
#include <iostream>
#include "tensors.h"
#include <algorithm>
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

float* Tensor::operator[](const size_t i) { return &data[i*width]; }

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result(height, other.getWidth());
    size_t B = 16;


    // Blocked loop with 16^3 = 4096 block size
    for (size_t II = 0; II < height; II += B) {
        for (size_t JJ = 0; JJ < other.getWidth(); JJ += B) {
            for (size_t KK = 0; KK < width; KK += B) {
                
                for (size_t i = II; i < min(II + B, height); i++) {
                    for (size_t k = KK; k < min(KK + B, width); k++) {

                        float x = data[i*width + k];
                        for (size_t j = JJ; j < min(JJ + B, other.getWidth()); j++) {
                            result[i][j] += x * other.data[k*other.getWidth() + j];
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
            result[i][j] = result[i][j] + other.data[i*width + j];
        }
    }


    return result;
}

void Tensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i*width + j] << " ";
        }
        cout << endl;
    }
}

