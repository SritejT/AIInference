#include "tensor.h"
#include "strategies/tensor_strategy.h"
#include "strategies/basic_simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"

class OptimisedTensorStrategy : public TensorStrategy {
private:
    static constexpr BasicSimdTensorStrategy simd_strategy = BasicSimdTensorStrategy();
    static constexpr ConcurrentRowTensorStrategy concurrent_strategy = ConcurrentRowTensorStrategy();

public:
    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;

    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override; 
};
