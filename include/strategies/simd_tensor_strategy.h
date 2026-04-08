#pragma once
#include "strategies/tensor_strategy.h"

class SimdTensorStrategy : public TensorStrategy {
    
public:

    void process_add_block(
            const Tensor* A, 
            const Tensor* B, 
            Tensor* result,
            size_t start_row,
            size_t start_col,
            size_t end_row,
            size_t end_col) const;

    void process_mult_block(
            const Tensor* A,
            const Tensor* B,
            Tensor* result,
            size_t start_row,
            size_t start_col,
            size_t end_row,
            size_t end_col) const;

    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override;
};
