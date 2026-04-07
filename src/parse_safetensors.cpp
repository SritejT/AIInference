#include <iostream>
#include <fstream>
#include <cstring>
#include "rapidjson/document.h"
#include "tensor.h"
#include "parse_safetensors.h"
#include "strategies/concurrent_row_tensor_strategy.h"


SafeTensorsParser::SafeTensorsParser(std::string fname) {

    f.open(fname, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Failed to open file: " << fname << std::endl;
        exit(1);
    }
}

std::string SafeTensorsParser::read_header_data() {
    f.read(reinterpret_cast<char*>(&header_size), 8);
    std::string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);
    return header_json;
}

std::vector<TensorInfo> SafeTensorsParser::parse_header(std::string header_json) {
    rapidjson::Document d;
    d.Parse(header_json.c_str());

    std::vector<TensorInfo> info;

    for (auto& m : d.GetObject()) {
        const std::string name = m.name.GetString();

        if (name == "__metadata__") continue;

        const std::string dtype = m.value["dtype"].GetString();

        const rapidjson::Value& shape = m.value["shape"];
        std::vector<size_t> shape_vec;
        
        for (rapidjson::SizeType i = 0; i < shape.Size(); i++) {
            shape_vec.push_back(shape[i].GetInt());
        }

        const rapidjson::Value& offsets = m.value["data_offsets"];
        size_t start = offsets[0].GetUint64();
        size_t end   = offsets[1].GetUint64();

        info.push_back({name, dtype, shape_vec, start, end});
    }

    return info;
}

std::vector<Tensor> SafeTensorsParser::parse_data(std::vector<TensorInfo>& info) {
    // Read tensors from safetensors file
    std::vector<Tensor> tensors;

    for (auto& t : info) {
        f.seekg(8 + header_size + t.start);
        std::vector<float> d((t.end - t.start) / sizeof(float));
        f.read(reinterpret_cast<char*>(&d[0]), t.end - t.start);

        tensors.push_back(Tensor(
                    d, 
                    t.shape[0], 
                    (t.shape.size() > 1 ? t.shape[1] : 1), 
                    std::make_shared<ConcurrentRowTensorStrategy>()));
    }

    return tensors; 
}

std::vector<Tensor> SafeTensorsParser::parse() {

    std::string header_json = read_header_data();
    std::vector<TensorInfo> info = parse_header(header_json);
    return parse_data(info);
}

SafeTensorsParser::~SafeTensorsParser() {
    f.close();
}

