#include <cstring>
#include "tensor.h"
#include "strategies/optimised_tensor_strategy.h"
#include <string>

int main(int argc, char** argv) {
    int n = std::stoi(std::string(argv[1]));
    std::vector<float> a = std::vector<float>(n * n, 1.0f);
    std::vector<float> b = std::vector<float>(n * n, 1.0f);

    TensorStrategy& strategy = OptimisedTensorStrategy::get_instance();

    Tensor A = Tensor(a, n, n, strategy);
    Tensor B = Tensor(b, n, n, strategy);

    Tensor result = A * B;

    return 0;
}
