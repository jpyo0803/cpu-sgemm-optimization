#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "custom_matmul.h"

namespace py = pybind11;

py::array_t<float> py_matmul_naive(py::array_t<float> A, py::array_t<float> B) {
  auto bufA = A.request();
  auto bufB = B.request();

  // 차원 검증
  if (bufA.ndim != 2 || bufB.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2-dimensional");
  }
  if (bufA.shape[1] != bufB.shape[0]) {
    throw std::runtime_error(
        "Matrix dimensions do not match for multiplication");
  }

  // 크기는 1024로 고정
  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  // 결과를 담을 배열 할당 및 0으로 초기화 (C++의 calloc처럼 동작)
  auto result = py::array_t<float>({M, N});
  auto bufResult = result.request();

  float* ptrA = static_cast<float*>(bufA.ptr);
  float* ptrB = static_cast<float*>(bufB.ptr);
  float* ptrResult = static_cast<float*>(bufResult.ptr);

  // 배열 메모리를 0으로 초기화 (+= 연산을 위해 필수)
  std::fill(ptrResult, ptrResult + (M * N), 0.0f);

  // 로우레벨 C++ 연산 호출
  MatmulImplNaive<M, N, K>(ptrA, ptrB, ptrResult);

  return result;
}

py::array_t<float> py_matmul_naive_register_acc(py::array_t<float> A,
                                                  py::array_t<float> B) {
  auto bufA = A.request();
  auto bufB = B.request();

  // 차원 검증
  if (bufA.ndim != 2 || bufB.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2-dimensional");
  }
  if (bufA.shape[1] != bufB.shape[0]) {
    throw std::runtime_error(
        "Matrix dimensions do not match for multiplication");
  }

  // 크기는 1024로 고정
  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  // 결과를 담을 배열 할당 및 초기화
  auto result = py::array_t<float>({M, N});
  auto bufResult = result.request();

  float* ptrA = static_cast<float*>(bufA.ptr);
  float* ptrB = static_cast<float*>(bufB.ptr);
  float* ptrResult = static_cast<float*>(bufResult.ptr);

  // 배열 메모리를 초기화 (register accumulator 버전은 직접 초기화)
  std::fill(ptrResult, ptrResult + (M * N), 0.0f);

  // 로우레벨 C++ 연산 호출
  MatmulImplNaiveRegisterAcc<M, N, K>(ptrA, ptrB, ptrResult);

  return result;
}

py::array_t<float> py_matmul_loop_order(py::array_t<float> A, py::array_t<float> B) {
  auto bufA = A.request();
  auto bufB = B.request();

  // 차원 검증
  if (bufA.ndim != 2 || bufB.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2-dimensional");
  }
  if (bufA.shape[1] != bufB.shape[0]) {
    throw std::runtime_error(
        "Matrix dimensions do not match for multiplication");
  }

  // 크기는 1024로 고정
  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;

  // 결과를 담을 배열 할당 및 초기화
  auto result = py::array_t<float>({M, N});
  auto bufResult = result.request();

  float* ptrA = static_cast<float*>(bufA.ptr);
  float* ptrB = static_cast<float*>(bufB.ptr);
  float* ptrResult = static_cast<float*>(bufResult.ptr);

  // 배열 메모리를 초기화
  std::fill(ptrResult, ptrResult + (M * N), 0.0f);

  // 로우레벨 C++ 연산 호출
  MatmulImplLoopOrder<M, N, K>(ptrA, ptrB, ptrResult);

  return result;
}

py::array_t<float> py_matmul_tiling_1d(py::array_t<float> A, py::array_t<float> B) {
  auto bufA = A.request();
  auto bufB = B.request();

  // 차원 검증
  if (bufA.ndim != 2 || bufB.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2-dimensional");
  }
  if (bufA.shape[1] != bufB.shape[0]) {
    throw std::runtime_error(
        "Matrix dimensions do not match for multiplication");
  }

  // 크기는 1024로 고정
  constexpr int M = 1024;
  constexpr int K = 1024;
  constexpr int N = 1024;
  constexpr int TileSize = 64; // 타일 크기

  // 결과를 담을 배열 할당 및 초기화
  auto result = py::array_t<float>({M, N});
  auto bufResult = result.request();

  float* ptrA = static_cast<float*>(bufA.ptr);
  float* ptrB = static_cast<float*>(bufB.ptr);
  float* ptrResult = static_cast<float*>(bufResult.ptr);

  // 배열 메모리를 초기화
  std::fill(ptrResult, ptrResult + (M * N), 0.0f);

  // 로우레벨 C++ 연산 호출
  MatmulImplTiling1D<M, N, K, TileSize>(ptrA, ptrB, ptrResult);

  return result;
}

PYBIND11_MODULE(custom_backend, m) {
  m.doc() = "Custom C++ CPU Matrix Multiplication Backend";
  m.def("matmul_naive", &py_matmul_naive, "Naive O(N^3) Matrix Multiplication");
  m.def("matmul_naive_register_acc", &py_matmul_naive_register_acc,
        "Naive Matrix Multiplication with Register Accumulator");
  m.def("matmul_loop_order", &py_matmul_loop_order,
      "Matrix Multiplication with Different Loop Order");
  m.def("matmul_tiling_1d", &py_matmul_tiling_1d,
        "Matrix Multiplication with 1D Tiling Optimization");
}