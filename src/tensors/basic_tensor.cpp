#include <vector>
#include "basic_tensor.h"

using namespace std;

BasicTensor::BasicTensor(size_t h, size_t w): ITensor(h, w) {}

BasicTensor::BasicTensor(vector<float> d, size_t h, size_t w): ITensor(d, h, w) {}

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


