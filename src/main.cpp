#include <cstring>
#include <memory>
#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"


int main() {
    
    std::vector<float> a = std::vector<float>(1048576, 1.0f);
    std::vector<float> b = std::vector<float>(1048576, 1.0f);

    Tensor A = Tensor(a, 1024, 1024, std::make_shared<BasicTensorStrategy>());
    Tensor B = Tensor(b, 1024, 1024, std::make_shared<BasicTensorStrategy>());

    Tensor result = A * B;

    return 0;
}
