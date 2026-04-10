#pragma once
#include "tensor.h"

class TensorStrategy {
private:

    virtual void subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const {

        size_t width = A->getWidth();

        for (size_t i = 0; i < width; i++) {
            A->data[row1 * width + i] -= multiple * A->data[row2 * width + i];
        }
    }

    virtual void scale_row(Tensor* A, size_t row, float multiple) const {
        size_t width = A->getWidth();
        for (size_t i = 0; i < width; i++) {
            A->data[row * width + i] *= multiple;
        }
    }


public:
    virtual void mult(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void add(const Tensor* A, const Tensor* B, Tensor* result) const = 0;
    virtual void transpose(const Tensor* A, Tensor* result) const = 0;

    virtual void inverse(Tensor* A, Tensor* result) const {

        size_t n = A->getHeight();

        for (size_t i = 0; i < n; i++) {
            result->data[i * n + i] = 1.0f;
        }

        for (size_t i = 0; i < n; i++) {
            float m1 = A->data[i * n + i];
            if (m1 <= 1e-10f && m1 >= -1e-10f) {
                throw std::runtime_error("Cannot invert a singular matrix");
            }
            scale_row(A, i, 1.0f / m1);
            scale_row(result, i, 1.0f / m1);

            for (size_t j = i+1; j < n; j++) {
                float m2 = A->data[j * n + i];
                subtract_rows(A, j, i, m2);
                subtract_rows(result, j, i, m2);
            }
        }

        for (int i = n-1; i >= 0; i--) {
            for (int j = i-1; j >= 0; j--) {
                float m3 = A->data[j * n + i];
                subtract_rows(A, j, i, m3);
                subtract_rows(result, j, i, m3);
            }
        }
    }

    virtual void apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const = 0;

    virtual ~TensorStrategy() = default;
};
