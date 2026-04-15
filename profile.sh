perf stat -e L1-dcache-loads,L1-dcache-load-misses,branch-instructions,branch-misses build/benchmarks/mult_benchmarks --benchmark_filter="<SimdTensorStrategy>"
perf stat -e L1-dcache-loads,L1-dcache-load-misses,branch-instructions,branch-misses build/benchmarks/mult_benchmarks --benchmark_filter="<BlockedSimdTensorStrategy>"
