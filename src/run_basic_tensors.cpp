#include <chrono>
#include <iostream>
#include <cstring>
#include "basic_tensor.h"
#include "parse_safetensors.h"

using namespace std;

int main() {
    
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();

    SafeTensorsParser parser("models/model.safetensors");
    
    vector<FastTensor> sustensors = parser.parse();

    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    auto parse_duration = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Parse Time: " << parse_duration << " ns" << endl;

    // Convert from Tensors to BasicTensors

    vector<BasicTensor> tensors;

    for (auto t : sustensors) {
        vector<float> d;
        for (auto f : t) {
            d.push_back(f);
        }
        tensors.push_back(BasicTensor(d, t.getHeight(), t.getWidth()));
    }
    

    // Test tensor addition and multiplication
    //
    t1 = chrono::high_resolution_clock::now();

    vector<float> input(784, 0.1f);
    BasicTensor input_tensor = BasicTensor(input, 784, 1);

    BasicTensor acc = tensors[1] * input_tensor;
    acc = acc + tensors[0];

    acc = tensors[3] * acc;
    acc = acc + tensors[2];

    acc = tensors[5] * acc;
    acc = acc + tensors[4];

    t2 = chrono::high_resolution_clock::now();
    auto math_duration = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << "Math Time: " << math_duration << " ns" << endl;

    return 0;
}
