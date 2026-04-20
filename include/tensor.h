#pragma once
#include <vector>
#include <functional>

class TensorStrategy; 

class Tensor {
private:
    size_t height, width;
    TensorStrategy& strategy;
    std::vector<float> data;

    Tensor apply(std::function<float(float)> f) const;

public:

    Tensor(size_t h, size_t w, TensorStrategy& strategy);
    Tensor(std::vector<float> d, size_t h, size_t w, TensorStrategy& strategy);


    size_t getWidth() const;
    size_t getHeight() const;

    std::vector<float>::const_iterator begin();
    std::vector<float>::const_iterator end();

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    Tensor operator+(const Tensor& other) const;
    
    Tensor operator*(float scalar) const;
    Tensor operator*(const Tensor& other) const;

    Tensor operator-() const;
    Tensor operator-(const Tensor& other) const;

    Tensor operator/(float scalar) const;
    
    Tensor transpose() const;
    Tensor inverse() const;

    void display() const;
};

