#pragma once
#include "tensor_strategy.h"

class ConcurrentRowTensorStrategy : public TensorStrategy {
private:
    void process_mult_rows(
            const Tensor* A,
            const Tensor* B, Tensor* result,
            size_t start_row,
            size_t end_row) const; 

    void process_add_rows(
            const Tensor* A,
            const Tensor* B, Tensor* result,
            size_t start_row,
            size_t end_row) const; 

public:

    void add(const Tensor* A, const Tensor* B, Tensor* result) const; 

    void mult(const Tensor* A, const Tensor* B, Tensor* result) const; 

};


