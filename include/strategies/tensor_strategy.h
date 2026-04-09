#pragma once
#include "tensor.h"

class TensorStrategy {
public:
    virtual void mult(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void add(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void transpose(const Tensor* A, Tensor* result) const = 0;

    virtual ~TensorStrategy() = default;
};
