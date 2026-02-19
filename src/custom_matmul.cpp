#include "custom_matmul.h"

void matmulImplNaive(const float *A, const float *B, float *result, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                result[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}