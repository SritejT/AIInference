#include "strategies/concurrent_row_tensor_strategy.h"
#include "tensor.h"

void ConcurrentRowTensorStrategy::add(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; i++) {

        auto fut = pool.submit([=, this]() {
            simd_strategy->process_add_block(
                A, 
                B,
                result,
                i * height / num_threads,
                0,
                (i + 1) * height / num_threads,
                B->getWidth()
            );
        });

        futures.push_back(std::move(fut));

    }

    for (auto& fut : futures) {
        fut.get();
    }

}

void ConcurrentRowTensorStrategy::mult(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::future<void>> futures;

    for (size_t i = 0; i < num_threads; i++) {

        auto fut = pool.submit([=, this]() {
            simd_strategy->process_mult_block(
                A, 
                B,
                result,
                i * height / num_threads,
                0,
                (i + 1) * height / num_threads,
                B->getWidth()
            );
        });

        futures.push_back(std::move(fut));

    }

    for (auto& fut : futures) {
        fut.get();
    }
}



