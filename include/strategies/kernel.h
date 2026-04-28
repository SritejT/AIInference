#pragma once
#include <arm_sve.h>
#include <memory>

/*
  GotoBLAS-style tiled GEMM kernel targeting ARM SVE.

  All matrices passed to these functions are in column-major order (or
  the custom block-packed format described per function).  The public
  Tensor class uses row-major storage; callers must convert before
  invoking these routines.

  Template parameters:
    mc  – number of rows per A / C panel (must divide m, or caller pads)
    kc  – K-strip width              (must divide k, or caller pads)
    nr  – number of columns per B / C micro-panel (must divide n, or caller pads)
*/
template<int mc, int kc, int nr>
class Kernel {

    // ── private constants ──────────────────────────────────────────────
    enum : int { FLOATS_PER_CACHE_LINE = 16 };

    // Second arg to __builtin_prefetch: 0 = read, 1 = write
    enum : int { K_READ = 0, K_WRITE = 1 };

    // Third arg to __builtin_prefetch: temporal locality (0–3)
    enum : int { K_L1 = 3, K_L2 = 2, K_L3 = 1 };

    // How many k-iterations ahead to prefetch A columns in mult_blocks.
    // Sized for a ~10–20 cycle L1 load-to-use latency on Cortex-X/A series.
    // Retune by changing this single constant.
    static constexpr int PREFETCH_K_DIST = 4;

public:

    /*
    Input Dimensions:
    A = [mc, kc]
    B = [kc, nr]
    C = [mc, nr]

    All three matrices are column-major (leading dimension = first
    template parameter of each, i.e. mc for A/C, kc for B).
    */
    static void mult_blocks(float* A, float* B, float* C) {
        int simd_width = svcntw();
        for (int i = 0; i < mc; i += simd_width) {
            for (int j = 0; j < nr; j++) {

                // Predicate: active lanes while (i + lane) < mc
                svbool_t pg = svwhilelt_b32(i, mc);

                // Prefetch next j-column of C into L1 while we compute
                // the current column, so the store target is hot by the
                // time svst1 executes on the next iteration.
                if (j + 1 < nr)
                    __builtin_prefetch(C + (j + 1) * mc + i, K_WRITE, K_L1);

                // Accumulate into C[i:i+simd_width, j]
                svfloat32_t sum = svld1_f32(pg, C + j * mc + i);
                for (int k = 0; k < kc; k++) {
                    // Prefetch an A column PREFETCH_K_DIST iterations ahead so
                    // the load latency is hidden inside the FMA stream.
                    if (k + PREFETCH_K_DIST < kc)
                        __builtin_prefetch(A + (k + PREFETCH_K_DIST) * mc + i, K_READ, K_L1);
                    svfloat32_t va = svld1_f32(pg, A + k * mc + i); // A[i:, k]
                    svfloat32_t vb = svdup_n_f32(B[j * kc + k]);    // broadcast B[k, j]
                    sum = svmla_f32_x(pg, sum, va, vb);
                }
                svst1_f32(pg, C + j * mc + i, sum);
            }
        }
    }


    /*
    Input Dimensions:
    A = [mc, kc]    column-major, ld = mc
    B = [kc, n]     column-major, ld = kc
    C = [mc, n]     column-major, ld = mc

    Computes C += A * B, partitioning n into nr-wide panels.
    */
    static void gebp(float* A, float* B, float* C, int n) {

        // Prefetch A panel into L2
        for (int i = 0; i < mc * kc; i += FLOATS_PER_CACHE_LINE) {
            __builtin_prefetch(A + i, K_READ, K_L2);
        }

        int N = n / nr;

        // ── Warm-up prefetch ──────────────────────────────────────────────
        // Issue the prefetch for the very first panel (or remainder when
        // N == 0) before entering the loop so that compute and memory are
        // already overlapping from iteration 0 onward.
        if (N > 0) {
            for (int j = 0; j < kc * nr; j += FLOATS_PER_CACHE_LINE)
                __builtin_prefetch(B + j, K_READ, K_L1);
            for (int j = 0; j < mc * nr; j += FLOATS_PER_CACHE_LINE)
                __builtin_prefetch(C + j, K_WRITE, K_L1);
        } else if (N * nr < n) {
            // n < nr: no full panels, only a remainder — prefetch it now.
            for (int j = 0; j < kc * n; j += FLOATS_PER_CACHE_LINE)
                __builtin_prefetch(B + j, K_READ, K_L1);
            for (int j = 0; j < mc * n; j += FLOATS_PER_CACHE_LINE)
                __builtin_prefetch(C + j, K_WRITE, K_L1);
        }

        for (int i = 0; i < N; i++) {
            // ── Look-ahead prefetch ───────────────────────────────────────
            // Issue the prefetch for the *next* panel at the top of the
            // current iteration so it overlaps with the mult_blocks call
            // below.  On the last full panel, prefetch the remainder instead
            // (if one exists); it was not yet prefetched by the warm-up.
            if (i + 1 < N) {
                for (int j = (i + 1) * kc * nr; j < (i + 2) * kc * nr; j += FLOATS_PER_CACHE_LINE)
                    __builtin_prefetch(B + j, K_READ, K_L1);
                for (int j = (i + 1) * mc * nr; j < (i + 2) * mc * nr; j += FLOATS_PER_CACHE_LINE)
                    __builtin_prefetch(C + j, K_WRITE, K_L1);
            } else if (N * nr < n) {
                // i == N-1 and a remainder exists: prefetch it while
                // mult_blocks runs for the final full panel.
                for (int j = N * kc * nr; j < kc * n; j += FLOATS_PER_CACHE_LINE)
                    __builtin_prefetch(B + j, K_READ, K_L1);
                for (int j = N * mc * nr; j < mc * n; j += FLOATS_PER_CACHE_LINE)
                    __builtin_prefetch(C + j, K_WRITE, K_L1);
            }

            mult_blocks(A, B + i * kc * nr, C + i * mc * nr);
        }

        // Handle the remainder columns when nr does not evenly divide n.
        // Guard is required: when n == N*nr the pointers below would be
        // one-past-the-end and mult_blocks would access out-of-bounds memory.
        // Data was already prefetched above (warm-up or look-ahead), so no
        // additional prefetch is needed here.
        if (N * nr < n) {
            mult_blocks(A, B + N * kc * nr, C + N * mc * nr);
        }
    }


    /*
    Input Dimensions:
    A = [m, kc]   column-major, ld = m
    B = [kc, n]   column-major, ld = kc  (already packed for gebp)
    C = [m, n]    block-packed: (m/mc) panels of [mc, n] each stored
                  column-major with ld = mc, laid out contiguously.
                  Panel i starts at C + i*mc*n.

    Computes C += A * B by partitioning rows into mc-strips.
    */
    static void gepp(float* A, float* B, float* C, int m, int n) {

        int Nblocks = m / mc;

        alignas(64) float A_pack[mc * kc];

        for (int i = 0; i < Nblocks; i++) {
            // Pack A[i*mc:(i+1)*mc, 0:kc] from ld=m into ld=mc
            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mc; r++) {
                    A_pack[k * mc + r] = A[k * m + i * mc + r];
                }
            }
            gebp(A_pack, B, C + i * mc * n, n);
        }

        // Remainder rows when mc does not evenly divide m
        if (Nblocks * mc < m) {
            for (int k = 0; k < kc; k++) {
                for (int r = 0; r < mc; r++) {
                    A_pack[k * mc + r] = A[k * m + Nblocks * mc + r];
                }
            }
            gebp(A_pack, B, C + Nblocks * mc * n, n);
        }
    }


    /*
    Input Dimensions:
    A = [m, k]   column-major, ld = m
    B = [k, n]   column-major, ld = k
    C = [m, n]   block-packed (see gepp comment above)

    Computes C += A * B in the Goto-paper style: partitions K into
    kc-wide strips, packs each B strip, and dispatches to gepp.

    C must be zero-initialised by the caller before the first call
    when a fresh C = A*B (rather than C += A*B) is desired.
    */
    static void gemm(float* A, float* B, float* C, int m, int k, int n) {

        int Kblocks = k / kc;

        std::unique_ptr<float[]> B_pack(new float[kc * n]);

        for (int pc = 0; pc < Kblocks; pc++) {
            // Pack B[pc*kc:(pc+1)*kc, 0:n] from ld=k into ld=kc
            for (int j = 0; j < n; j++) {
                for (int q = 0; q < kc; q++) {
                    B_pack[j * kc + q] = B[j * k + pc * kc + q];
                }
            }
            gepp(A + pc * kc * m, B_pack.get(), C, m, n);
        }

        // Remainder K-strip when kc does not evenly divide k
        if (Kblocks * kc < k) {
            for (int j = 0; j < n; j++) {
                for (int q = 0; q < kc; q++) {
                    B_pack[j * kc + q] = B[j * k + Kblocks * kc + q];
                }
            }
            gepp(A + Kblocks * kc * m, B_pack.get(), C, m, n);
        }
    }

};
