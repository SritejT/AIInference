#pragma once
#include <cstddef>
#include <strategies/tensor_strategy.h>

class SimdTensorStrategy : public TensorStrategy {
protected:
    void process_mult_block(
            const Tensor* A,
            const Tensor* B,
            Tensor* result,
            size_t start_row,
            size_t start_col,
            size_t end_row,
            size_t end_col) const; 

    void process_add_block(
            const Tensor* A,
            const Tensor* B,
            Tensor* result,
            size_t start_row,
            size_t start_col,
            size_t end_row,
            size_t end_col) const; 

public:
    virtual void add(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void mult(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
}; 
