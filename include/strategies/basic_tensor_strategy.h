#pragma once
#include "strategies/tensor_strategy.h"

class BasicTensorStrategy : public TensorStrategy {
private:
    BasicTensorStrategy() = default;
public:

    // Only make one instance that the whole program shares
    static BasicTensorStrategy& get_instance() {
        static BasicTensorStrategy instance;
        return instance;
    }

    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void transpose(const Tensor* A, Tensor* result) const override;

    void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const override;

    BasicTensorStrategy(const BasicTensorStrategy&) = delete;
    void operator=(const BasicTensorStrategy&) = delete;

    BasicTensorStrategy(BasicTensorStrategy&&) = delete;
    void operator=(BasicTensorStrategy&&) = delete;
};
