#include "itensor.h"
#include <vector>
#include <iostream>

using namespace std;

ITensor::ITensor(vector<float> d, size_t h, size_t w) {
    width = w;
    height = h;
    data = d;
}

ITensor::ITensor(size_t h, size_t w) {
    width = w;
    height = h;
    data.resize(h * w, 0.0f);
}

size_t ITensor::getWidth() const {
    return width;
}

size_t ITensor::getHeight() const {
    return height;
}

vector<float>::const_iterator ITensor::begin() {
    return data.begin();
}

vector<float>::const_iterator ITensor::end() {
    return data.end();
}

void ITensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i * width + j] << " ";
        }
        cout << endl;
    }
}
