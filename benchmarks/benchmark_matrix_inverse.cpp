#include "strategies/basic_tensor_strategy.h"
#include "strategies/simd_tensor_strategy.h"
#include "strategies/concurrent_row_tensor_strategy.h"
#include "strategies/optimised_tensor_strategy.h"
#include "strategies/blocked_simd_tensor_strategy.h"

#include <memory>
#include <vector>
#include <benchmark/benchmark.h>

template<typename Strategy>
static void TensorSquareMatInverse(benchmark::State& state) {

    int n = state.range(0);
    auto strategy = std::make_shared<Strategy>();

    std::vector<float> a(n * n, 0.0f);
    for (int i=0; i<n; i++) {
        a[i*n + ((i + 1) % n)] = 1.0f;
    }

    Tensor A = Tensor(a, n, n, strategy);
    
    for (auto _ : state) {
        Tensor B = A.inverse();
        benchmark::DoNotOptimize(B);
    }
}

BENCHMARK(TensorSquareMatInverse<BasicTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatInverse<SimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatInverse<ConcurrentRowTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatInverse<OptimisedTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);
BENCHMARK(TensorSquareMatInverse<BlockedSimdTensorStrategy>)->RangeMultiplier(2)->Range(2, 1024);

BENCHMARK_MAIN();

