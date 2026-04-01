#pragma once
#include <vector>
using namespace std;

class Tensor {
protected:
    size_t height, width;
    Tensor(size_t h, size_t w);
    Tensor(vector<float> d, size_t h, size_t w);

public:

    vector<float> data;

    size_t getWidth() const;
    size_t getHeight() const;

    vector<float>::const_iterator begin();
    vector<float>::const_iterator end();

    void display() const;
};
