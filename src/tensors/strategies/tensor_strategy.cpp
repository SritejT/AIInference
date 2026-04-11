#include "strategies/tensor_strategy.h"


void TensorStrategy::subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const {

    size_t width = A->getWidth();

    for (size_t i = 0; i < width; i++) {
        A->data[row1 * width + i] -= multiple * A->data[row2 * width + i];
    }
}

void TensorStrategy::scale_row(Tensor* A, size_t row, float multiple) const {
    size_t width = A->getWidth();
    for (size_t i = 0; i < width; i++) {
        A->data[row * width + i] *= multiple;
    }
}

void TensorStrategy::swap_rows(Tensor* A, size_t row1, size_t row2) const {
    size_t width = A->getWidth();
    for (size_t i = 0; i < width; i++) {
        std::swap(A->data[row1 * width + i], A->data[row2 * width + i]);
    }
}

// We use Gaussian elimination
void TensorStrategy::inverse(const Tensor* A, Tensor* result) const {

    Tensor A_copy = *A;
    size_t n = A->getHeight();

    // Initialise RHS matrix to identity
    for (size_t i = 0; i < n; i++) {
        result->data[i * n + i] = 1.0f;
    }

    for (size_t i = 0; i < n; i++) {

        // If diagonal element is 0, look for a row below that we can swap it with
        float m1 = A_copy.data[i * n + i];
        if (m1 <= 1e-10f && m1 >= -1e-10f) {

            for (size_t j = i+1; j < n; j++) {
                if (A_copy.data[j * n + i] > 1e-10f || A_copy.data[j * n + i] < -1e-10f) {

                    swap_rows(&A_copy, i, j);
                    swap_rows(result, i, j);

                    m1 = A_copy.data[i * n + i];
                    break;
                }
            }
        }


        // If the diagonal element is still 0, we know the matrix is 0
        if (m1 <= 1e-10f && m1 >= -1e-10f) {
            throw std::runtime_error("Cannot invert a singular matrix");
        }

        // Normalise row so that the diagonal element is 1
        scale_row(&A_copy, i, 1.0f / m1);
        scale_row(result, i, 1.0f / m1);

        // Subtract row i from all other rows so that all elements below the diagonal element are 0
        for (size_t j = i+1; j < n; j++) {
            float m2 = A_copy.data[j * n + i];
            subtract_rows(&A_copy, j, i, m2);
            subtract_rows(result, j, i, m2);
        }
    }

    // Now we are in row echelon form
    // Use a similar process as above to eliminate elements above the diagonal
    for (int i = n-1; i >= 0; i--) {
        for (int j = i-1; j >= 0; j--) {
            float m3 = A_copy.data[j * n + i];
            subtract_rows(&A_copy, j, i, m3);
            subtract_rows(result, j, i, m3);
        }
    }
}
