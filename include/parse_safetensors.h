#include <fstream>
#include <cstring>
#include "tensors.h"

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

    string read_header_data();

    vector<TensorInfo> parse_header(string header_json);

    vector<Tensor> parse_data(vector<TensorInfo>& info); 


public:
    
    SafeTensorsParser(string fname);

    ~SafeTensorsParser();

    vector<Tensor> parse(); 
};

