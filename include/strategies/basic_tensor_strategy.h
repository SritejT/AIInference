#pragma once
#include "strategies/tensor_strategy.h"

class BasicTensorStrategy : public TensorStrategy {
public:
    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void transpose(const Tensor* A, Tensor* result) const override;

    void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const override;
};
