#include "tensor.h"
#include <vector>
#include <iostream>
#include "strategies/tensor_strategy.h"

using namespace std;

Tensor::Tensor(vector<float> d, size_t h, size_t w, shared_ptr<TensorStrategy> s) :
    width(w), height(h), data(d), strategy(s) {

    if (d.size() != h * w || h <= 0 || w <= 0) {
        throw runtime_error("Invalid tensor size");
    }

}

Tensor::Tensor(size_t h, size_t w, shared_ptr<TensorStrategy> s) : 
    width(w), height(h), strategy(s) {

    if (h <= 0 || w <= 0) {
        throw runtime_error("Invalid tensor size");
    }

    width = w;
    height = h;
    data.resize(h * w);
}

size_t Tensor::getWidth() const {
    return width;
}

size_t Tensor::getHeight() const {
    return height;
}

vector<float>::const_iterator Tensor::begin() {
    return data.begin();
}

vector<float>::const_iterator Tensor::end() {
    return data.end();
}

Tensor Tensor::operator+(const Tensor& other) const {
    if (width != other.width || height != other.height) {
        throw runtime_error("Invalid tensor sizes");
    }

    Tensor result(height, width, strategy);
    for (size_t i = 0; i < height * width; i++) {
        result.data[i] = data[i] + other.data[i];
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (width != other.height) {
        throw runtime_error("Invalid tensor sizes");
    }

    Tensor result(height, other.width, strategy);
    strategy->mult(this, &other, &result);
    return result;

}

void Tensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i * width + j] << " ";
        }
        cout << endl;
    }
}
