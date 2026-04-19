set -e
 
cmake -S . -B build -DENABLE_TSAN=OFF -DENABLE_ASAN_UBSAN=ON -DENABLE_MSAN=OFF -DENABLE_BENCHMARKS=OFF
cmake --build build --parallel
build/tests/all_tests 
