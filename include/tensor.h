#pragma once
#include <vector>
#include <memory>

using namespace std;

class TensorStrategy; 

class Tensor {
protected:
    size_t height, width;
    shared_ptr<TensorStrategy> strategy;

public:

    Tensor(size_t h, size_t w, shared_ptr<TensorStrategy> strategy);
    Tensor(vector<float> d, size_t h, size_t w, shared_ptr<TensorStrategy> strategy);

    vector<float> data;

    size_t getWidth() const;
    size_t getHeight() const;

    vector<float>::const_iterator begin();
    vector<float>::const_iterator end();

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    void display() const;
};

