#include "tensor.h"
#include <vector>
#include <iostream>
#include "strategies/tensor_strategy.h"

Tensor::Tensor(std::vector<float> d, size_t h, size_t w, std::shared_ptr<TensorStrategy> s) :
    width(w), height(h), data(d), strategy(s) {

    if (d.size() != h * w || h <= 0 || w <= 0) {
        throw std::runtime_error("Invalid tensor size");
    }

}

Tensor::Tensor(size_t h, size_t w, std::shared_ptr<TensorStrategy> s) : 
    width(w), height(h), strategy(s) {

    if (h <= 0 || w <= 0) {
        throw std::runtime_error("Invalid tensor size");
    }

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

std::vector<float>::const_iterator Tensor::begin() {
    return data.begin();
}

std::vector<float>::const_iterator Tensor::end() {
    return data.end();
}

Tensor Tensor::apply(std::function<float(float)> f) const {
    Tensor result(height, width, strategy);
    strategy->apply(f, this, &result);
    return result;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if ((width != other.width) || (height != other.height)) {
        throw std::runtime_error("Invalid tensor sizes");
    }

    Tensor result(height, width, strategy);
    strategy->add(this, &other, &result); 
    return result;
}

Tensor Tensor::operator-() const {
    return this->apply([&](float data) { return -data; });
}

Tensor Tensor::operator-(const Tensor& other) const {

    if ((width != other.width) || (height != other.height)) {
        throw std::runtime_error("Invalid tensor sizes");
    }

    const Tensor negated = -other;

    return *this + negated;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (width != other.height) {
        throw std::runtime_error("Invalid tensor sizes");
    }

    Tensor result(height, other.width, strategy);
    strategy->mult(this, &other, &result);
    return result;

}

Tensor Tensor::operator*(float scalar) const {
    return this->apply([scalar](float x) { return x * scalar; });
}

Tensor Tensor::operator/(float scalar) const {
    return this->apply([scalar](float x) { return x / scalar; });
}

Tensor Tensor::transpose() const {
    Tensor result(width, height, strategy);
    strategy->transpose(this, &result);
    return result;
}

Tensor Tensor::inverse() {
    if (width != height) {
        throw std::runtime_error("Cannot invert a non-square matrix");
    }

    Tensor result(width, height, strategy);
    strategy->inverse(this, &result);
    return result;
}

void Tensor::display() const {
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            std::cout << data[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}


