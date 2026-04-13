#pragma once
#include "strategies/tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "threadpool.h"

class ConcurrentRowTensorStrategy : public TensorStrategy {

private:
    inline static Threadpool& pool = Threadpool::get_instance();
    inline static std::shared_ptr<SimdTensorStrategy> simd_strategy = std::make_shared<SimdTensorStrategy>();

public:

    ConcurrentRowTensorStrategy() = default;

    void swap_rows(Tensor* A, size_t row1, size_t row2) const override;
    void subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const override;
    void scale_row(Tensor* A, size_t row, float multiple) const override;

    void add(const Tensor* A, const Tensor* B, Tensor* result) const override; 
    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override; 
    void transpose(const Tensor* A, Tensor* result) const override;

    void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const override;

};


