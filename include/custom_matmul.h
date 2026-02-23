#pragma once

template <int M, int N, int K>
inline void MatmulImplNaive(const float *A, const float *B, float *result) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        result[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

template <int M, int N, int K>
inline void MatmulImplNaiveRegisterAcc(const float *A, const float *B,
                                       float *result) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += A[m * K + k] * B[k * N + n];
      }
      result[m * N + n] = acc;
    }
  }
}

template <int M, int N, int K>
inline void MatmulImplLoopOrder(const float *A, const float *B, float *result) {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        result[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}