#include <benchmark/benchmark.h>
#include <vector>
#include <memory>
#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/concurrent_blocked_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"

template <typename Strategy>
static void TensorSquareMatMul(benchmark::State& state) {
    int n = state.range(0);

    auto strategy = std::make_shared<Strategy>();

    Tensor a(std::vector<float>(n * n, 1.0f), n, n, strategy);
    Tensor b(std::vector<float>(n * n, 1.0f), n, n, strategy);
    for (auto _ : state) {
        Tensor c = a * b;
        benchmark::DoNotOptimize(c);
    }
}

template <typename Strategy>
static void TensorMatxVecMul(benchmark::State& state) {
    int n = state.range(0);

    auto strategy = std::make_shared<Strategy>();

    Tensor a(std::vector<float>(n * n, 1.0f), n, n, strategy);
    Tensor b(std::vector<float>(n, 1.0f), n, 1, strategy);

    for (auto _ : state) {
        Tensor c = a * b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(TensorSquareMatMul<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatMul<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatMul<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatMul<ConcurrentBlockedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorSquareMatMul<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);

BENCHMARK(TensorMatxVecMul<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecMul<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecMul<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecMul<ConcurrentBlockedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);
BENCHMARK(TensorMatxVecMul<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 2048);

BENCHMARK_MAIN();
