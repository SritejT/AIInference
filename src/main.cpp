#include <cstring>
#include <memory>
#include "tensor.h"
#include "strategies/concurrent_row_tensor_strategy.h"


int main() {
    int n = 1025;
    std::vector<float> a = std::vector<float>(n * n, 1.0f);
    std::vector<float> b = std::vector<float>(n * n, 1.0f);

    Tensor A = Tensor(a, n, n, std::make_shared<ConcurrentRowTensorStrategy>());
    Tensor B = Tensor(b, n, n, std::make_shared<ConcurrentRowTensorStrategy>());

    Tensor result = A * B;

    return 0;
}
