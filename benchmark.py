import numpy as np
import time
import custom_matmul as cmm

def benchmark(func, A, B):
    # Warm-up
    _ = func(A, B)

    min_latency = float('inf')

    for _ in range(20):
        start_time = time.perf_counter()
        result = func(A, B)
        end_time = time.perf_counter()
        min_latency = min(min_latency, end_time - start_time)

    return result, min_latency

def main():
    # 튜토리얼과 동일하게 1024x1024 사이즈 생성
    N = 1024
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    print(f"Matrix size: {N}x{N}")

    # Numpy (내부적으로 고도로 최적화된 OpenBLAS/MKL 사용)
    C_numpy, numpy_time = benchmark(lambda A, B: A @ B, A, B)
    print(f"NumPy (BLAS) Time: {numpy_time:.4f} seconds")

    # Custom C++ (Naive)
    C_custom_naive, custom_time = benchmark(cmm.CustomMatMulNaive().matmul, A, B)
    print(f"Custom Naive C++ Time: {custom_time:.4f} seconds, Speedup: {numpy_time / custom_time:.4f}x")

    # Custom C++ (Register Accumulation)
    C_custom_register, custom_register_time = benchmark(cmm.CustomMatMulRegisterAcc().matmul, A, B)
    print(f"Custom Register Accumulation C++ Time: {custom_register_time:.4f} seconds, Speedup: {numpy_time / custom_register_time:.4f}x")
    
    # Custom C++ (Loop Order)
    C_custom_loop_order, custom_loop_order_time = benchmark(cmm.CustomMatMulLoopOrder().matmul, A, B)
    print(f"Custom Loop Order C++ Time: {custom_loop_order_time:.4f} seconds, Speedup: {numpy_time / custom_loop_order_time:.4f}x")
    
    # Custom C++ (Tiling)
    C_custom_tiling_1d, custom_tiling_1d_time = benchmark(cmm.CustomMatMulTiling1D().matmul, A, B)
    print(f"Custom Tiling 1D C++ Time: {custom_tiling_1d_time:.4f} seconds, Speedup: {numpy_time / custom_tiling_1d_time:.4f}x")
    
    # Custom C++ (Row-Column Parallel Tiling 1D)
    C_custom_row_col_parallel_tiling_1d, custom_row_col_parallel_tiling_1d_time = benchmark(cmm.CustomMatMulRowColParallelTiling1D().matmul, A, B)
    print(f"Custom Row-Column Parallel Tiling 1D C++ Time: {custom_row_col_parallel_tiling_1d_time:.4f} seconds, Speedup: {numpy_time / custom_row_col_parallel_tiling_1d_time:.4f}x")

    # 결과 검증 (오차 범위 내에서 같은지 확인)
    np.testing.assert_allclose(C_custom_naive, C_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C_custom_register, C_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C_custom_loop_order, C_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C_custom_tiling_1d, C_numpy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C_custom_row_col_parallel_tiling_1d, C_numpy, rtol=1e-4, atol=1e-4)
    print("Success: Results match")

if __name__ == "__main__":
    main()