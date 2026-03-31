#include <benchmark/benchmark.h>
#include <vector>
#include "fast_tensor.h"

using namespace std;

static void BM_SquareMatMul(benchmark::State& state) {
    int n = state.range(0);
    FastTensor a(vector<float>(n * n, 1.0f), n, n);
    FastTensor b(vector<float>(n * n, 1.0f), n, n);
    for (auto _ : state) {
        FastTensor c = a * b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(BM_SquareMatMul)->Arg(64)->Arg(256)->Arg(1024);
BENCHMARK_MAIN();
