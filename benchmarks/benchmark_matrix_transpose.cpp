#include <benchmark/benchmark.h>
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

template <typename Strategy>
static void TensorSquareMatTranspose(benchmark::State& state) {

    int n = state.range(0);
    auto& strategy = Strategy::get_instance();

    Tensor a(std::vector<float>(n * n, 1.0f), n, n, &strategy);
    for (auto _ : state) {
        Tensor c = a.transpose();
        benchmark::DoNotOptimize(c);
    }
}

template <typename Strategy>
static void TensorVecTranspose(benchmark::State& state) {

    int n = state.range(0);
    auto& strategy = Strategy::get_instance();

    Tensor a(std::vector<float>(n, 1.0f), n, 1, &strategy);

    for (auto _ : state) {
        Tensor c = a.transpose();
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(TensorSquareMatTranspose<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatTranspose<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatTranspose<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatTranspose<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatTranspose<BlockedSimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);

BENCHMARK(TensorVecTranspose<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorVecTranspose<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorVecTranspose<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorVecTranspose<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorVecTranspose<BlockedSimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);

BENCHMARK_MAIN();
