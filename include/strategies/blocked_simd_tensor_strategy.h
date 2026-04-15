#include <arm_neon.h>
#include "strategies/tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"

class BlockedSimdTensorStrategy : public TensorStrategy {
private:
    inline static SimdTensorStrategy simd_strategy = SimdTensorStrategy();

    void subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const override;
    void scale_row(Tensor* A, size_t row, float multiple) const override;
    void swap_rows(Tensor* A, size_t row1, size_t row2) const override;

public:

    void mult(const Tensor* A, const Tensor* B, Tensor* result) const override;
    void add(const Tensor* A, const Tensor* B, Tensor* result) const override;

    void transpose(const Tensor* A, Tensor* result) const override;
    void inverse(const Tensor* A, Tensor* result) const override; 

    void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const override;

};
