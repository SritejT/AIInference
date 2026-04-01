set -e
 
cmake -S . -B build
cmake --build build
ctest --test-dir build/tests --output-on-failure
./build/AIInference
