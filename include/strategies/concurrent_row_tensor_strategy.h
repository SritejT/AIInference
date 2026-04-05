#pragma once
#include "strategies/simd_tensor_strategy.h"

class ConcurrentRowTensorStrategy : public SimdTensorStrategy {

public:

    void add(const Tensor* A, const Tensor* B, Tensor* result) const; 
    void mult(const Tensor* A, const Tensor* B, Tensor* result) const; 

};


