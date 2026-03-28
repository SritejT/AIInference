#include <vector>
#include <iostream>

#include "basic_tensors.h"

using namespace std;

BasicTensor::BasicTensor(size_t h, size_t w) {
    width = w;
    height = h;
    data.resize(h * w); 
}

BasicTensor::BasicTensor(vector<float> d, size_t h, size_t w) {
    width = w;
    height = h;
    data = d;
}

size_t BasicTensor::getWidth() const { return width; }
size_t BasicTensor::getHeight() const { return height; }

BasicTensor BasicTensor::operator*(const BasicTensor& other) const {
    BasicTensor result(height, other.getWidth());

    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < other.getWidth(); j++) {
            
            float sum = 0.0f;
            for (size_t k = 0; k < width; k++) {
                sum += data[i * width + k] * other.data[k * other.getWidth() + j];
            }
            result.data[i * other.getWidth() + j] = sum;
            
        }
    }

    return result;
}

BasicTensor BasicTensor::operator+(const BasicTensor& other) const {
    BasicTensor result(height, width);

    for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i]; 
    }

    return result;
}

vector<float>::const_iterator BasicTensor::begin() const { return data.begin(); }
vector<float>::const_iterator BasicTensor::end() const { return data.end(); }

void BasicTensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            cout << data[i * width + j] << " ";
        }
        cout << endl;
    }
}
