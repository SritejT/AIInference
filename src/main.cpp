#include <cstring>
#include <memory>
#include "tensor.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"


int main() {
    int n = 1024;
    std::vector<float> a = std::vector<float>(n * n, 1.0f);
    std::vector<float> b = std::vector<float>(n * n, 1.0f);

    Tensor A = Tensor(a, n, n, std::make_shared<BlockedSimdTensorStrategy>());
    Tensor B = Tensor(b, n, n, std::make_shared<BlockedSimdTensorStrategy>());

    Tensor result = A * B;

    return 0;
}
