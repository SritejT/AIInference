#include <vector>
#include <iostream>
#include "tensors.h"
using namespace std;

Tensor::Tensor(size_t h, size_t w) {
    width = w;
    height = h;

    data.resize(h);
    for (size_t i = 0; i < h; i++) {
        data[i].resize(w);
    }
}

Tensor::Tensor(vector<float> d, size_t h, size_t w) {
    width = w;
    height = h;

    data.resize(h);
    for (size_t i = 0; i < h; i++) {
        data[i].resize(w);
        for (size_t j = 0; j < w; j++) {
            data[i][j] = d[i * w + j];
        }
    }
}

size_t Tensor::getWidth() const { return width; }
size_t Tensor::getHeight() const { return height; }

Tensor Tensor::operator*(const Tensor& other) const {
    Tensor result(this->getHeight(), other.getWidth());
    for (size_t i = 0; i < this->getHeight(); i++) {
        for (size_t j = 0; j < other.getWidth(); j++) {
            float sum = 0;
            for (size_t k = 0; k < this->getWidth(); k++) {
                sum += data[i][k] * other.data[k][j];
            }
            result.data[i][j] = sum;
        }
    }
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    Tensor result(this->getHeight(), this->getWidth());
    for (size_t i = 0; i < this->getHeight(); i++) {
        for (size_t j = 0; j < this->getWidth(); j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

void Tensor::display() const {
    for (size_t i = 0; i < this->getHeight(); i++) {
        for (size_t j = 0; j < this->getWidth(); j++) {
            cout << data[i][j] << " ";
        }
        cout << endl;
    }
}

