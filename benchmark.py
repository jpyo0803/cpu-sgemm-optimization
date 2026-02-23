import numpy as np
import time
import custom_matmul as cmm

# 튜토리얼과 동일하게 1024x1024 사이즈 생성
N = 1024
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Warm-up
_ = cmm.CustomMatMulNaive().matmul(A, B)
_ = cmm.CustomMatMulRegisterAcc().matmul(A, B)
_ = cmm.CustomMatMulLoopOrder().matmul(A, B)
_ = cmm.CustomMatMulTiling1D().matmul(A, B)
_ = A @ B

print(f"Matrix size: {N}x{N}")

# Custom C++ (Naive)
C_custom_naive, custom_time = cmm.CustomMatMulNaive().matmul(A, B)
print(f"Custom Naive C++ Time: {custom_time:.4f} seconds")

# Custom C++ (Register Accumulation)
C_custom_register, custom_register_time = cmm.CustomMatMulRegisterAcc().matmul(A, B)
print(f"Custom Register Accumulation C++ Time: {custom_register_time:.4f} seconds")

# Custom C++ (Loop Order)
C_custom_loop_order, custom_loop_order_time = cmm.CustomMatMulLoopOrder().matmul(A, B)
print(f"Custom Loop Order C++ Time: {custom_loop_order_time:.4f} seconds")

# Custom C++ (Tiling)
C_custom_tiling_1d, custom_tiling_1d_time = cmm.CustomMatMulTiling1D().matmul(A, B)
print(f"Custom Tiling 1D C++ Time: {custom_tiling_1d_time:.4f} seconds")

# Numpy (내부적으로 고도로 최적화된 OpenBLAS/MKL 사용)
start = time.time()
C_numpy = A @ B  # 또는 np.dot(A, B)
end = time.time()
numpy_time = end - start
print(f"NumPy (BLAS) Time: {numpy_time:.4f} seconds")

# 결과 검증 (오차 범위 내에서 같은지 확인)
np.testing.assert_allclose(C_custom_naive, C_numpy, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(C_custom_register, C_numpy, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(C_custom_loop_order, C_numpy, rtol=1e-4, atol=1e-4)
np.testing.assert_allclose(C_custom_tiling_1d, C_numpy, rtol=1e-4, atol=1e-4)
print("Success: Results match")