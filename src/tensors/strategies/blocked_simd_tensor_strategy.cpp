#include "strategies/blocked_simd_tensor_strategy.h"
#include "strategies/kernel.h"
#include <vector>

#define BLOCK_SIZE 8

// Template parameters for Kernel<mc, kc, nr>.
// All three must evenly divide the padded matrix dimensions — inputs are
// zero-padded up to the next multiple of each value before calling gemm.
static constexpr int KMC = 32;
static constexpr int KKC = 32;
static constexpr int KNR = 32;

static inline int round_up(int val, int mult) {
    return ((val + mult - 1) / mult) * mult;
}

void BlockedSimdTensorStrategy::subtract_rows(Tensor* A, size_t row1, size_t row2, float multiple) const {
    simd_strategy.subtract_rows(A, row1, row2, multiple);
}

void BlockedSimdTensorStrategy::scale_row(Tensor* A, size_t row, float multiple) const {
    simd_strategy.scale_row(A, row, multiple);
}

void BlockedSimdTensorStrategy::swap_rows(Tensor* A, size_t row1, size_t row2) const {
    simd_strategy.swap_rows(A, row1, row2);
}

void BlockedSimdTensorStrategy::add(const Tensor* A, const Tensor* B, Tensor* result) const {
    process_add_block(A, B, result, 0, 0, result->getHeight(), result->getWidth());
}

void BlockedSimdTensorStrategy::process_add_block(
        const Tensor* A, 
        const Tensor* B, 
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t end_row,
        size_t end_col) const {
    simd_strategy.process_add_block(A, B, result, start_row, start_col, end_row, end_col);
}

void BlockedSimdTensorStrategy::mult(const Tensor* A, const Tensor* B, Tensor* result) const {
    const int m = static_cast<int>(A->getHeight());
    const int k = static_cast<int>(A->getWidth());   // == B->getHeight()
    const int n = static_cast<int>(B->getWidth());

    // Pad each dimension to the next multiple of the corresponding kernel
    // block parameter so that all remainder paths inside the kernel are
    // never reached (avoids out-of-bounds reads/writes in gepp/gemm).
    const int m_p = round_up(m, KMC);
    const int k_p = round_up(k, KKC);
    const int n_p = round_up(n, KNR);

    // A_col: column-major [m_p, k_p], zero-padded
    std::vector<float> A_col(m_p * k_p, 0.0f);
    // B_col: column-major [k_p, n_p], zero-padded
    std::vector<float> B_col(k_p * n_p, 0.0f);
    // C_blk: block-packed [m_p, n_p], zero-initialised
    // Layout: (m_p/KMC) row-panels; panel i at C_blk + i*KMC*n_p;
    // within a panel column j is at offset j*KMC, row r at j*KMC+r.
    std::vector<float> C_blk(m_p * n_p, 0.0f);

    // Convert A from row-major [m, k] to column-major [m_p, k_p]
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A_col[j * m_p + i] = (*A)[i * k + j];
        }
    }

    // Convert B from row-major [k, n] to column-major [k_p, n_p]
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B_col[j * k_p + i] = (*B)[i * n + j];
        }
    }

    // Run the kernel: C_blk += A_col * B_col (C_blk is zero so this is C = A*B)
    Kernel<KMC, KKC, KNR>::gemm(A_col.data(), B_col.data(), C_blk.data(), m_p, k_p, n_p);

    // Unpack block-packed C_blk back into row-major result [m, n].
    // Element [i, j] lives at:
    //   block_row  = i / KMC,  row_in_block  = i % KMC
    //   panel_col  = j / KNR,  col_in_panel  = j % KNR
    //   C_blk[ block_row*KMC*n_p + panel_col*KMC*KNR + col_in_panel*KMC + row_in_block ]
    for (int i = 0; i < m; i++) {
        const int block_row    = i / KMC;
        const int row_in_block = i % KMC;
        for (int j = 0; j < n; j++) {
            const int panel_col   = j / KNR;
            const int col_in_panel = j % KNR;
            (*result)[i * n + j] = C_blk[
                block_row    * KMC * n_p
              + panel_col    * KMC * KNR
              + col_in_panel * KMC
              + row_in_block
            ];
        }
    }
}

void BlockedSimdTensorStrategy::process_mult_block(
        const Tensor* A,
        const Tensor* B,
        Tensor* result,
        size_t start_row,
        size_t start_col,
        size_t start_k,
        size_t end_row,
        size_t end_col,
        size_t end_k) const {

    size_t result_height = A->getHeight();
    size_t result_width = B->getWidth();

    size_t A_width = A->getWidth(); 


    for (size_t I = start_row; I < end_row; I += BLOCK_SIZE) {
        for (size_t J = start_col; J < end_col; J += BLOCK_SIZE) {
            for (size_t K = start_k; K < end_k; K += BLOCK_SIZE) {
                simd_strategy.process_mult_block(
                        A,
                        B, 
                        result,
                        I, 
                        J, 
                        K, 
                        std::min(I + BLOCK_SIZE, end_row),
                        std::min(J + BLOCK_SIZE, end_col),
                        std::min(K + BLOCK_SIZE, end_k));
            }
        }
    }
}
void BlockedSimdTensorStrategy::transpose(const Tensor* A, Tensor* result) const {
    simd_strategy.process_transpose_block(A, result, 0, A->getHeight());
}

void BlockedSimdTensorStrategy::process_transpose_block(
        const Tensor* A,
        Tensor* result,
        size_t start_row,
        size_t end_row) const {
    simd_strategy.process_transpose_block(A, result, start_row, end_row);
}

void BlockedSimdTensorStrategy::inverse(const Tensor* A, Tensor* result) const {
    simd_strategy.inverse(A, result);
}

void BlockedSimdTensorStrategy::apply(std::function<float(float)> f, const Tensor* A, Tensor* result) const {
    process_apply_block(f, A, result, 0, A->getHeight());
}

void BlockedSimdTensorStrategy::process_apply_block(std::function<float(float)> f, const Tensor* A, Tensor* result, size_t start_row, size_t end_row) const {
    simd_strategy.process_apply_block(f, A, result, start_row, end_row);
}
