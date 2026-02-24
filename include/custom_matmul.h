#pragma once

template <int M, int N, int K>
inline void MatmulImplNaive(const float* A, const float* B, float* result) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        result[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

template <int M, int N, int K>
inline void MatmulImplNaiveRegisterAcc(const float* A, const float* B,
                                       float* result) {
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
inline void MatmulImplLoopOrder(const float* A, const float* B, float* result) {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        result[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

template <int M, int N, int K, int TileSize>
inline void MatmulImplTiling1D(const float* A, const float* B, float* result) {
  for (int tile_offset = 0; tile_offset < K; tile_offset += TileSize) {
    for (int m = 0; m < M; ++m) {
      int tile_offset_end = std::min(K, tile_offset + TileSize);
      for (int k = tile_offset; k < tile_offset_end; ++k) {
        for (int n = 0; n < N; ++n) {
          result[m * N + n] += A[m * K + k] * B[k * N + n];
        }
      }
    }
  }
}

template <int M, int N, int K, int TileSize>
inline void MatmulImplTiling2D(const float* A, const float* B, float* result) {
  for (int tile_offset1 = 0; tile_offset1 < M; tile_offset1 += TileSize) {
    for (int tile_offset2 = 0; tile_offset2 < K; tile_offset2 += TileSize) {
      for (int tile_offset3 = 0; tile_offset3 < N; tile_offset3 += TileSize) {
        for (int m = tile_offset1; m < tile_offset1 + TileSize; ++m) {
          for (int k = tile_offset2; k < tile_offset2 + TileSize; ++k) {
            for (int n = tile_offset3; n < tile_offset3 + TileSize; ++n) {
              result[m * N + n] += A[m * K + k] * B[k * N + n];
            }
          }
        }
      }
    }
  }
}

template <int M, int N, int K, int TileSize>
inline void MatmulImplRowColParallelTiling1D(const float* A, const float* B,
                                             float* result) {
#pragma omp parallel for shared(result, A, B) default(none) collapse(2) \
    num_threads(8)

  for (int row_tile = 0; row_tile < M; row_tile += 256) {
    for (int col_tile = 0; col_tile < N; col_tile += 256) {
      for (int inner_tile = 0; inner_tile < K; inner_tile += TileSize) {
        for (int m = row_tile; m < row_tile + 256; ++m) {
          int inner_tile_end = std::min(K, inner_tile + TileSize);
          for (int k = inner_tile; k < inner_tile_end; ++k) {
            for (int n = col_tile; n < col_tile + 256; ++n) {
              result[m * N + n] += A[m * K + k] * B[k * N + n];
            }
          }
        }
      }
    }
  }
}