#include <cstring>
#include "tensor.h"
#include "strategies/blocked_simd_tensor_strategy.h"


int main() {
    int n = 1024;
    std::vector<float> a = std::vector<float>(n * n, 1.0f);
    std::vector<float> b = std::vector<float>(n * n, 1.0f);

    TensorStrategy& strategy = BlockedSimdTensorStrategy::get_instance();

    Tensor A = Tensor(a, n, n, &strategy);
    Tensor B = Tensor(b, n, n, &strategy);

    Tensor result = A * B;

    return 0;
}
