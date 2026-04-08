set -e
 
cmake -S . -B build
cmake --build build
build/tests/all_tests 
