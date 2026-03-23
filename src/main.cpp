#include <cstdint>
#include <iostream>
#include <fstream>
#include <cstring>
#include "rapidjson/document.h"
#include "tensors.h"

using namespace std;
using namespace rapidjson;

struct TensorInfo {
    string name;
    string dtype;
    vector<size_t> shape;
    size_t start;
    size_t end;
};


int main() {
    ifstream f("src/model.safetensors", ios::binary);
    if (!f) {
        cout << "Error opening file" << endl;
        return -1;
    }

    // Read json header of safetensors file
    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    cout << "Header size: " << header_size << endl;

    string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);

    cout << "Header: " << header_json << endl;

    // Parse json header
    Document d;
    d.Parse(header_json.c_str());
    

    vector<TensorInfo> info;


    for (auto& m : d.GetObject()) {
        const string name = m.name.GetString();

        if (name == "__metadata__") continue;

        const string dtype = m.value["dtype"].GetString();

        const Value& shape = m.value["shape"];
        vector<size_t> shape_vec;
        
        for (SizeType i = 0; i < shape.Size(); i++) {
            shape_vec.push_back(shape[i].GetInt());
        }

        const Value& offsets = m.value["data_offsets"];
        size_t start = offsets[0].GetUint64();
        size_t end   = offsets[1].GetUint64();

        info.push_back({name, dtype, shape_vec, start, end});
    }
    
    // Read tensors from safetensors file
    vector<Tensor> tensors;

    for (auto& t : info) {
        f.seekg(t.start);
        vector<float> d((t.end - t.start) / sizeof(float));
        f.read(reinterpret_cast<char*>(&d[0]), t.end - t.start);

        tensors.push_back(Tensor(d, t.shape[0], (
                        t.shape.size() > 1 ? t.shape[1] : 1
                        )
                    ));
    }

    vector<float> input(784, 0.1f);
    Tensor input_tensor = Tensor(input, 784, 1);

    Tensor acc = tensors[1] * input_tensor;
    acc = acc + tensors[0];

    acc = tensors[3] * acc;
    acc = acc + tensors[2];

    acc = tensors[5] * acc;
    acc = acc + tensors[4];

    acc.display();

    f.close();

    return 0;
}
