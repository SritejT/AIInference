#pragma once
#include <fstream>
#include <cstring>
#include "fast_tensor.h"

using namespace std;

struct TensorInfo {
    string name;
    string dtype;
    vector<size_t> shape;
    size_t start;
    size_t end;
};

class SafeTensorsParser {
private:

    ifstream f;
    uint64_t header_size;

    string read_header_data();

    vector<TensorInfo> parse_header(string header_json);

    vector<FastTensor> parse_data(vector<TensorInfo>& info); 


public:
    
    SafeTensorsParser(string fname);

    ~SafeTensorsParser();

    vector<FastTensor> parse(); 
};

