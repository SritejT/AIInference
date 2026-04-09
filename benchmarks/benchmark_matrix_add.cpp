#include <benchmark/benchmark.h>
#include <vector>
#include "tensor.h"
#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"

template <typename Strategy>
static void TensorSquareMatAdd(benchmark::State& state) {
    int n = state.range(0);

    auto strategy = std::make_shared<Strategy>();

    Tensor a(std::vector<float>(n * n, 1.0f), n, n, strategy);
    Tensor b(std::vector<float>(n * n, 1.0f), n, n, strategy);

    for (auto _ : state) {
        Tensor c = a + b;
        benchmark::DoNotOptimize(c);
    }
}

BENCHMARK(TensorSquareMatAdd<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatAdd<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatAdd<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatAdd<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);

BENCHMARK_MAIN();
