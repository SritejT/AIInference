#pragma once
#include <vector>
using namespace std;

class ITensor {
protected:
    vector<float> data;
    size_t height, width;
    ITensor(size_t h, size_t w);
    ITensor(vector<float> d, size_t h, size_t w);

public:

    size_t getWidth() const;
    size_t getHeight() const;

    vector<float>::const_iterator begin();
    vector<float>::const_iterator end();

    void display() const;
};
