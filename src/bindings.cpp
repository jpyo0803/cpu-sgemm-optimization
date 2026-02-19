#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "custom_matmul.h"
#include <stdexcept>

namespace py = pybind11;

py::array_t<float> py_matmul_naive(py::array_t<float> A, py::array_t<float> B) {
    auto bufA = A.request();
    auto bufB = B.request();

    // 차원 검증
    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::runtime_error("Input arrays must be 2-dimensional");
    }
    if (bufA.shape[1] != bufB.shape[0]) {
        throw std::runtime_error("Matrix dimensions do not match for multiplication");
    }

    // 크기 추출
    int M = bufA.shape[0];
    int K = bufA.shape[1];
    int N = bufB.shape[1];

    // 결과를 담을 배열 할당 및 0으로 초기화 (C++의 calloc처럼 동작)
    auto result = py::array_t<float>({M, N});
    auto bufResult = result.request();
    
    float* ptrA = static_cast<float*>(bufA.ptr);
    float* ptrB = static_cast<float*>(bufB.ptr);
    float* ptrResult = static_cast<float*>(bufResult.ptr);

    // 배열 메모리를 0으로 초기화 (+= 연산을 위해 필수)
    std::fill(ptrResult, ptrResult + (M * N), 0.0f);

    // 로우레벨 C++ 연산 호출
    matmulImplNaive(ptrA, ptrB, ptrResult, M, N, K);

    return result;
}

PYBIND11_MODULE(custom_backend, m) {
    m.doc() = "Custom C++ CPU Matrix Multiplication Backend";
    m.def("matmul_naive", &py_matmul_naive, "Naive O(N^3) Matrix Multiplication");
}