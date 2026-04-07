#pragma once
#include <fstream>
#include <cstring>
#include "tensor.h"

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<size_t> shape;
    size_t start;
    size_t end;
};

class SafeTensorsParser {
private:

    std::ifstream f;
    uint64_t header_size;

    std::string read_header_data();

    std::vector<TensorInfo> parse_header(std::string header_json);

    std::vector<Tensor> parse_data(std::vector<TensorInfo>& info); 


public:
    
    SafeTensorsParser(std::string fname);

    ~SafeTensorsParser();

    std::vector<Tensor> parse(); 
};

