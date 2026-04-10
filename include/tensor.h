#pragma once
#include <vector>
#include <memory>
#include <functional>

class TensorStrategy; 

class Tensor {
private:
    size_t height, width;
    std::shared_ptr<TensorStrategy> strategy;

    Tensor apply(std::function<float(float)> f) const;

public:

    Tensor(size_t h, size_t w, std::shared_ptr<TensorStrategy> strategy);
    Tensor(std::vector<float> d, size_t h, size_t w, std::shared_ptr<TensorStrategy> strategy);

    std::vector<float> data;

    size_t getWidth() const;
    size_t getHeight() const;

    std::vector<float>::const_iterator begin();
    std::vector<float>::const_iterator end();

    Tensor operator+(const Tensor& other) const;
    
    Tensor operator*(float scalar) const;
    Tensor operator*(const Tensor& other) const;

    Tensor operator-() const;
    Tensor operator-(const Tensor& other) const;

    Tensor operator/(float scalar) const;
    
    Tensor transpose() const;

    void display() const;
};

