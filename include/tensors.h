#pragma once
#include <vector>

using namespace std;

class Tensor {
private:
    vector<float> data;
    size_t height, width;
public:
    Tensor(size_t h, size_t w);
    Tensor(vector<float> d, size_t h, size_t w);

    size_t getWidth() const;
    size_t getHeight() const;
    
    float* operator[](const size_t i);
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(const Tensor& other) const;
    void display() const;
};
