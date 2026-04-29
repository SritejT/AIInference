perf stat -e cycles,instructions,\
cache-references,cache-misses,\
dTLB-loads,dTLB-load-misses \
./build/AIInference 2047

