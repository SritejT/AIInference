#include <iostream>
#include <fstream>
#include <cstring>
#include "rapidjson/document.h"
#include "tensor.h"
#include "parse_safetensors.h"
#include "strategies/concurrent_row_tensor_strategy.h"

using namespace std;
using namespace rapidjson;

SafeTensorsParser::SafeTensorsParser(string fname) {

    f.open(fname, ios::binary);
    if (!f.is_open()) {
        cerr << "Failed to open file: " << fname << endl;
        exit(1);
    }
}

string SafeTensorsParser::read_header_data() {
    f.read(reinterpret_cast<char*>(&header_size), 8);
    string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);
    return header_json;
}

vector<TensorInfo> SafeTensorsParser::parse_header(string header_json) {
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

    return info;
}

vector<Tensor> SafeTensorsParser::parse_data(vector<TensorInfo>& info) {
    // Read tensors from safetensors file
    vector<Tensor> tensors;

    for (auto& t : info) {
        f.seekg(8 + header_size + t.start);
        vector<float> d((t.end - t.start) / sizeof(float));
        f.read(reinterpret_cast<char*>(&d[0]), t.end - t.start);

        tensors.push_back(Tensor(
                    d, 
                    t.shape[0], 
                    (t.shape.size() > 1 ? t.shape[1] : 1), 
                    make_shared<ConcurrentRowTensorStrategy>()));
    }

    return tensors; 
}

vector<Tensor> SafeTensorsParser::parse() {

    string header_json = read_header_data();
    vector<TensorInfo> info = parse_header(header_json);
    return parse_data(info);
}

SafeTensorsParser::~SafeTensorsParser() {
    f.close();
}

