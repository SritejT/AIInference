#include "tensor.h"
#include "strategies/tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"

class OptimisedTensorStrategy : public TensorStrategy {
private:
    inline static std::shared_ptr<SimdTensorStrategy> simd_strategy = std::make_shared<SimdTensorStrategy>();
    inline static std::shared_ptr<ConcurrentRowTensorStrategy> concurrent_strategy = std::make_shared<ConcurrentRowTensorStrategy>();

public:
    OptimisedTensorStrategy() = default;

    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;

    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override; 
};
