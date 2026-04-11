#pragma once
#include "tensor.h"

class TensorStrategy {
private:

    virtual void subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const; 

    virtual void scale_row(Tensor* A, size_t row, float multiple) const; 

    virtual void swap_rows(Tensor* A, size_t row1, size_t row2) const;

public:
    virtual void mult(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void add(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void transpose(const Tensor* A, Tensor* result) const = 0;

    virtual void inverse(const Tensor* A, Tensor* result) const; 

    virtual void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const = 0;

    virtual ~TensorStrategy() = default;
};
