#pragma once
#include "strategies/tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "tensor.h"
#include "threadpool.h"

class ConcurrentBlockedTensorStrategy: public TensorStrategy {
private:
    inline static Threadpool& pool = Threadpool::get_instance();
    inline static std::shared_ptr<SimdTensorStrategy> simd_strategy = std::make_shared<SimdTensorStrategy>();

public:

    void add(
            const Tensor *A,
            const Tensor *B,
            Tensor *result) const override; 

    void mult(
            const Tensor *A,
            const Tensor *B,
            Tensor *result) const override; 

    void transpose(
            const Tensor *A,
            Tensor *result) const override;
};
