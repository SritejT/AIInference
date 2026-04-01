#include <benchmark/benchmark.h>
#include <vector>
#include "fast_tensor.h"
#include "basic_tensor.h"

using namespace std;

static void BM_FastTensorSquareMatAdd(benchmark::State& state) {
    int n = state.range(0);
    FastTensor a(vector<float>(n * n, 1.0f), n, n);
    FastTensor b(vector<float>(n * n, 1.0f), n, n);
    for (auto _ : state) {
        FastTensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

static void BM_BasicTensorSquareMatAdd(benchmark::State& state) {
    int n = state.range(0);
    BasicTensor a(vector<float>(n * n, 1.0f), n, n);
    BasicTensor b(vector<float>(n * n, 1.0f), n, n);
    for (auto _ : state) {
        BasicTensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

static void BM_FastTensorMatxVecAdd(benchmark::State& state) {
    int n = state.range(0);
    FastTensor a(vector<float>(n * n, 1.0f), n, n);
    FastTensor b(vector<float>(n, 1.0f), n, 1);
    for (auto _ : state) {
        FastTensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

static void BM_BasicTensorMatxVecAdd(benchmark::State& state) {
    int n = state.range(0);
    BasicTensor a(vector<float>(n * n, 1.0f), n, n);
    BasicTensor b(vector<float>(n, 1.0f), n, 1);
    for (auto _ : state) {
        BasicTensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(BM_FastTensorSquareMatAdd)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_BasicTensorSquareMatAdd)->RangeMultiplier(2)->Range(64, 1024);

BENCHMARK(BM_FastTensorMatxVecAdd)->RangeMultiplier(2)->Range(64, 1024);
BENCHMARK(BM_BasicTensorMatxVecAdd)->RangeMultiplier(2)->Range(64, 1024);

BENCHMARK_MAIN();
