#include <chrono>
#include <iostream>
#include <cstring>
#include "tensors.h"
#include "parse_safetensors.h"

using namespace std;

int main() {
    
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    SafeTensorsParser parser("model.safetensors");
    
    vector<Tensor> tensors = parser.parse();
    
    // Test tensor addition and multiplication

    vector<float> input(784, 0.1f);
    Tensor input_tensor = Tensor(input, 784, 1);

    Tensor acc = tensors[1] * input_tensor;
    acc = acc + tensors[0];

    acc = tensors[3] * acc;
    acc = acc + tensors[2];

    acc = tensors[5] * acc;
    acc = acc + tensors[4];

    acc.display();

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Time: " << duration << " ns" << endl;
    return 0;
}
