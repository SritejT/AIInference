#include "strategies/concurrent_blocked_tensor_strategy.h"
#include <thread>
#include <cmath>

void ConcurrentBlockedTensorStrategy::add(
        const Tensor *A,
        const Tensor *B,
        Tensor *result) const {

    size_t height = result->getHeight();
    size_t width = result->getWidth();

    size_t num_cores = std::thread::hardware_concurrency();

    size_t grid_dim = sqrt(num_cores);

    std::vector<std::future<void>> futures;

    for (int i=0; i < grid_dim * grid_dim; i++) {
        size_t grid_i = i / grid_dim;
        size_t grid_j = i % grid_dim;

        size_t start_row = grid_i * height / grid_dim;
        size_t end_row = (grid_i + 1) * height / grid_dim;
        size_t start_col = grid_j * width / grid_dim;
        size_t end_col = (grid_j + 1) * width / grid_dim;

        auto fut = pool->submit([=, this]() {
            simd_strategy->process_add_block(
                A,
                B,
                result,
                start_row,
                start_col,
                end_row,
                end_col
            );
        }); 
           
        futures.push_back(std::move(fut));
    }
for (auto& fut : futures) {
        fut.get();
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

    std::vector<std::future<void>> futures;

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

        auto fut = pool->submit([=, this]() {
            simd_strategy->process_mult_block(
                A,
                B,
                result,
                start_row,
                start_col,
                end_row,
                end_col
            );
        });

        futures.push_back(std::move(fut));
    }
    
    for (auto& fut : futures) {
        fut.get();
    }
}
