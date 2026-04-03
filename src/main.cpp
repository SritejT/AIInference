#include <cstring>
#include "fast_tensor.h"

using namespace std;

int main() {
    
    vector<float> a = vector<float>(1048576, 1.0f);
    vector<float> b = vector<float>(1048576, 1.0f);

    FastTensor A = FastTensor(a, 1024, 1024);
    FastTensor B = FastTensor(b, 1024, 1024);

    FastTensor result = A * B;

    return 0;
}
