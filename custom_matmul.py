import custom_backend
import time
import numpy as np

class CustomMatMulBase:
    def __init__(self):
        pass

    def matmul(self, A: np.ndarray, B: np.ndarray):
        raise NotImplementedError("Subclasses should implement this method")

class CustomMatMulNaive(CustomMatMulBase):
    def matmul(self, A: np.ndarray, B: np.ndarray):
        # For simplicity, we assume A and B are 1024x1024 matrices
        assert A.shape[0] == 1024
        assert A.shape[1] == 1024
        assert B.shape[0] == 1024
        assert B.shape[1] == 1024

        start_time = time.time()
        result = custom_backend.matmul_naive(A, B)
        end_time = time.time()
        return result, end_time - start_time

class CustomMatMulRegisterAcc(CustomMatMulBase):
    def matmul(self, A: np.ndarray, B: np.ndarray):
        # For simplicity, we assume A and B are 1024x1024 matrices
        assert A.shape[0] == 1024
        assert A.shape[1] == 1024
        assert B.shape[0] == 1024
        assert B.shape[1] == 1024

        start_time = time.time()
        result = custom_backend.matmul_naive_register_acc(A, B)
        end_time = time.time()
        return result, end_time - start_time

class CustomMatMulLoopOrder(CustomMatMulBase):
    def matmul(self, A: np.ndarray, B: np.ndarray):
        # For simplicity, we assume A and B are 1024x1024 matrices
        assert A.shape[0] == 1024
        assert A.shape[1] == 1024
        assert B.shape[0] == 1024
        assert B.shape[1] == 1024

        start_time = time.time()
        result = custom_backend.matmul_loop_order(A, B)
        end_time = time.time()
        return result, end_time - start_time