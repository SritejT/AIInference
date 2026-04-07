#include "strategies/concurrent_row_tensor_strategy.h"
#include <arm_neon.h>
#include <thread>
#include <vector>

void ConcurrentRowTensorStrategy::add(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(std::thread(
            &ConcurrentRowTensorStrategy::process_add_block,
            this,
            A, 
            B,
            result,
            i * height / num_threads,
            0,
            (i + 1) * height / num_threads,
            B->getWidth()
        ));

    }

    for (auto& t : threads) {
        t.join();
    }
}

void ConcurrentRowTensorStrategy::mult(
        const Tensor* A,
        const Tensor* B,
        Tensor* result) const {

    size_t height = result->getHeight();

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; i++) {

        threads.push_back(std::thread(
            &ConcurrentRowTensorStrategy::process_mult_block,
            this,
            A, 
            B,
            result,
            i * height / num_threads,
            0,
            (i + 1) * height / num_threads,
            B->getWidth()
        ));

    }

    for (auto& t : threads) {
        t.join();
    }
}



