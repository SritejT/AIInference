#include <arm_sve.h>
#define FLOATS_PER_CACHE_LINE 16

enum modes {
    READ = 0,
    WRITE = 1,
};

enum localities {
    L1 = 3,
    L2 = 2,
    L3 = 1,
};


template<int mc, int kc, int nr>
class Kernel {
public: 

    /*
    Input Dimensions:
    A = [mc, kc]
    B = [kc, nr]
    C = [mc, nr]

    Assume matrices are packed and stored in column-major order
    */
    static void mult_blocks(float* A, float* B, float* C) {
        int simd_width = svcntw();
        for (int i=0; i<mc; i+=simd_width) {
            for (int j=0; j<nr; j++) {

                // Create predicate for this iteration
                svbool_t pg = svwhilelt_b32(i, mc);

                // Sum accumulates the value of C[i:i+simd_width][j]
                svfloat32_t sum = svld1_f32(pg, C+j*mc+i);
                for (int k=0; k<kc; k++) {
                    svfloat32_t va = svld1_f32(pg, A+k*mc+i); // Load A[i:i+simd_width][k]
                    svfloat32_t vb = svdup_n_f32(B[j*kc+k]); // Broadcast B[k][j]
                    sum = svmla_f32_x(pg, sum, va, vb); // Fused multiply-add
                }
                svst1_f32(pg, C+j*mc+i, sum); // Store back the result
            }            
        }        
    }


    /*
    Input Dimensions:
    A = [mc, kc]
    B = [kc, n]
    C = [mc, n]

    Assume matrices are stored in column major order to make B_j and C_j blocks contiguous in memory
    */
    static void gebp(float* A, float* B, float* C, int n) {

        for (int i = 0; i < mc*kc; i+=FLOATS_PER_CACHE_LINE) {
            __builtin_prefetch(A+i, READ, L2);
        }

        // Number of blocks that we split B and C into
        int N = n / nr;

        for (int i=0; i<N; i++) {

            // Prefetch B and C sub-matrices
            for (int j=i*kc*nr; j<(i+1)*kc*nr; j+=FLOATS_PER_CACHE_LINE) __builtin_prefetch(B+j, READ, L1);

            for (int j=i*mc*nr; j<(i+1)*mc*nr; j+=FLOATS_PER_CACHE_LINE) __builtin_prefetch(C+j, WRITE, L1);

            mult_blocks(A, B+i*kc*nr, C+i*mc*nr);
        }

        // Deal with the case when nr does not cleanly divide n
        for (int j=N*kc*nr; j<kc*n; j+=FLOATS_PER_CACHE_LINE) __builtin_prefetch(B+j, READ, L1);

        for (int j=N*mc*nr; j<mc*n; j+=FLOATS_PER_CACHE_LINE) __builtin_prefetch(C+j, WRITE, L1);

        mult_blocks(A, B+N*kc*nr, C+N*mc*nr);
    }

};
