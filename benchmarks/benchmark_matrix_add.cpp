#include <benchmark/benchmark.h>
#include <vector>
#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include "strategies/basic_simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/concurrent_blocked_tensor_strategy.h"

using namespace std;

template <typename Strategy>
static void TensorSquareMatAdd(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(vector<float>(n * n, 1.0f), n, n, make_shared<Strategy>());
    Tensor b(vector<float>(n * n, 1.0f), n, n, make_shared<Strategy>());
    for (auto _ : state) {
        Tensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

template <typename Strategy>
static void TensorMatxVecAdd(benchmark::State& state) {
    int n = state.range(0);
    Tensor a(vector<float>(n * n, 1.0f), n, n, make_shared<Strategy>());
    Tensor b(vector<float>(n, 1.0f), n, 1, make_shared<Strategy>());
    for (auto _ : state) {
        Tensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(TensorSquareMatAdd<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatAdd<BasicSimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatAdd<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatAdd<ConcurrentBlockedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);

BENCHMARK(TensorMatxVecAdd<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecAdd<BasicSimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecAdd<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecAdd<ConcurrentBlockedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);

BENCHMARK_MAIN();
