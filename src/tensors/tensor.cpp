#include "tensor.h"
#include <vector>
#include <iostream>

using namespace std;

Tensor::Tensor(vector<float> d, size_t h, size_t w) {
    width = w;
    height = h;
    data = d;
}

Tensor::Tensor(size_t h, size_t w) {
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

void Tensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i * width + j] << " ";
        }
        cout << endl;
    }
}
