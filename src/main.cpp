#include <cstring>
#include <memory>
#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"

using namespace std;

int main() {
    
    vector<float> a = vector<float>(1048576, 1.0f);
    vector<float> b = vector<float>(1048576, 1.0f);

    Tensor A = Tensor(a, 1024, 1024, make_shared<BasicTensorStrategy>());
    Tensor B = Tensor(b, 1024, 1024, make_shared<BasicTensorStrategy>());

    Tensor result = A * B;

    return 0;
}
