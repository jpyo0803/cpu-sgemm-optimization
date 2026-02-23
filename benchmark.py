import numpy as np
import time
import custom_backend

# 튜토리얼과 동일하게 1024x1024 사이즈 생성
N = 1024
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# Warm-up
_ = custom_backend.matmul_naive(A, B)
_ = custom_backend.matmul_naive_register_acc(A, B)
_ = A @ B

print(f"Matrix size: {N}x{N}")

# Custom C++ (Naive)
start = time.time()
C_custom = custom_backend.matmul_naive(A, B)
end = time.time()
custom_time = end - start
print(f"Custom Naive C++ Time: {custom_time:.4f} seconds")

# Custom C++ (Register Accumulation)
start = time.time()
C_custom = custom_backend.matmul_naive_register_acc(A, B)
end = time.time()
custom_register_time = end - start
print(f"Custom Register Accumulation C++ Time: {custom_register_time:.4f} seconds")

# Numpy (내부적으로 고도로 최적화된 OpenBLAS/MKL 사용)
start = time.time()
C_numpy = A @ B  # 또는 np.dot(A, B)
end = time.time()
numpy_time = end - start
print(f"NumPy (BLAS) Time: {numpy_time:.4f} seconds")

# 결과 검증 (오차 범위 내에서 같은지 확인)
np.testing.assert_allclose(C_custom, C_numpy, rtol=1e-4, atol=1e-4)
print("Success: Results match")