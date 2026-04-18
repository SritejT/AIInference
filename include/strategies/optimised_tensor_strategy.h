#include "tensor.h"
#include "strategies/tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"

class OptimisedTensorStrategy : public TensorStrategy {
private:
    inline static SimdTensorStrategy& simd_strategy = SimdTensorStrategy::get_instance();
    inline static ConcurrentRowTensorStrategy& concurrent_strategy = ConcurrentRowTensorStrategy::get_instance();

    OptimisedTensorStrategy() = default;

public:

    static OptimisedTensorStrategy& get_instance() {
        static OptimisedTensorStrategy instance;
        return instance;
    }

    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override; 

    void transpose(const Tensor* A, Tensor* result) const override;
    void inverse(const Tensor* A, Tensor* result) const override;

    void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const override; 

    OptimisedTensorStrategy(const OptimisedTensorStrategy&) = delete;
    void operator=(const OptimisedTensorStrategy&) = delete;

    OptimisedTensorStrategy(OptimisedTensorStrategy&&) = delete;
    void operator=(OptimisedTensorStrategy&&) = delete;
};
