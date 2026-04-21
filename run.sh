set -e
 
cmake -S . -B build -DENABLE_TSAN=OFF -DENABLE_ASAN_UBSAN=OFF -DENABLE_BENCHMARKS=ON
cmake --build build --parallel
build/tests/all_tests 
