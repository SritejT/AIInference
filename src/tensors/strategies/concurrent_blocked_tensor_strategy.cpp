#include "strategies/concurrent_blocked_tensor_strategy.h"
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <cmath>

void ConcurrentBlockedTensorStrategy::add(
        const Tensor *A,
        const Tensor *B,
        Tensor *result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    size_t num_cores = std::thread::hardware_concurrency();

    size_t grid_dim = sqrt(num_cores);

    std::vector<std::thread> threads;

    for (int i=0; i < grid_dim * grid_dim; i++) {
        size_t grid_i = i / grid_dim;
        size_t grid_j = i % grid_dim;

        size_t start_row = grid_i * height / grid_dim;
        size_t end_row = (grid_i + 1) * height / grid_dim;
        size_t start_col = grid_j * width / grid_dim;
        size_t end_col = (grid_j + 1) * width / grid_dim;

        threads.push_back(std::thread(
            &ConcurrentBlockedTensorStrategy::process_add_block,
            this,
            A,
            B,
            result,
            start_row,
            start_col,
            end_row,
            end_col
        ));
    }        

    for (auto& t : threads) {
        t.join();
    }
}

void ConcurrentBlockedTensorStrategy::mult(
        const Tensor *A,
        const Tensor *B,
        Tensor *result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    size_t num_cores = std::thread::hardware_concurrency();

    size_t grid_dim = sqrt(num_cores);

    std::vector<std::thread> threads;

    for (int i=0; i < grid_dim * grid_dim; i++) {
        size_t grid_i = i / grid_dim;
        size_t grid_j = i % grid_dim;

        size_t start_row = grid_i * height / grid_dim;
        size_t end_row = (grid_i + 1) * height / grid_dim;
        size_t start_col = (grid_j * width / grid_dim) / 4 * 4;

        size_t end_col; 
        if (grid_j + 1 == grid_dim) {
            end_col = width;
        } else {
            end_col = (grid_j + 1) * width / grid_dim;
        }

        threads.push_back(std::thread(
            &ConcurrentBlockedTensorStrategy::process_mult_block,
            this,
            A,
            B,
            result,
            start_row,
            start_col,
            end_row,
            end_col
        ));
    }        

    for (auto& t : threads) {
        t.join();
    }
}
