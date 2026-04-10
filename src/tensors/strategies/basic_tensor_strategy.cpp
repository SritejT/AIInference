#include "strategies/basic_tensor_strategy.h"
#include "tensor.h"

void BasicTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {
    
    size_t result_height = A->getHeight();
    size_t result_width = B->getWidth();

    size_t A_width = A->getWidth();

    for (size_t i = 0; i < result_height; i++) {
        for (size_t j = 0; j < result_width; j++) {
            
            float sum = 0.0f;
            for (size_t k = 0; k < A_width; k++) {
                sum += A->data[i * A_width + k] * B->data[k * result_width + j];
            }

            result->data[i * result_width + j] = sum;
            
        }
    }
}

void BasicTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {
    for (size_t i = 0; i < A->getHeight() * A->getWidth(); i++) {
        result->data[i] = A->data[i] + B->data[i];
    }
}

void BasicTensorStrategy::transpose(const Tensor* A, Tensor* result) const {
    for (size_t i = 0; i < A->getHeight(); i++) {
        for (size_t j = 0; j < A->getWidth(); j++) {
            result->data[j * A->getHeight() + i] = A->data[i * A->getWidth() + j];
        }
    }
}

void BasicTensorStrategy::apply(std::function<float(float)> f, Tensor* A, Tensor* result) const {
    for (size_t i = 0; i < A->getHeight() * A->getWidth(); i++) {
        result->data[i] = f(A->data[i]);
    }
}

