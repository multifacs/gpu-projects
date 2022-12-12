#ifndef _GPU_MATMUL_H_
#define _GPU_MATMUL_H_

// #define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define THREADS 12
#define BLOCK 16

#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>

#include "CL/cl.h"

#define CHECK_CORRECT(flag, type, reference, result, size)                \
  if (flag) {                                                     \
    bool res = checkCorrect<type>(result, reference, size);       \
    std::string result = "NO ERRORS";                                  \
    if (res == 0) result = "ERRORS FOUND";                               \
    std::cout << result << std::endl; \
  }

#define TIME_MS(t0, t1) std::chrono::duration_cast<ms>(t1 - t0).count() << " milliseconds"

using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using s = std::chrono::seconds;
using timer = std::pair<std::chrono::high_resolution_clock::time_point,
                        std::chrono::high_resolution_clock::time_point>;

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

void matrix_multiplication(const size_t m, const size_t n, const size_t k, const float* a,
            const float* b, float* c);
void matrix_multiplication_omp(const size_t m, const size_t n, const size_t k, const float* a,
                const float* b, float* c);
void matrix_multiplication_gpu(const size_t m, const size_t n, const size_t k, const float* a,
               const float* b, float* c,
               std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

void gemm(const size_t m, const size_t n, const size_t k, const float* a,
             const float* b, float* c,
             std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

template <typename T>
void createData(T* data, const size_t size) {
  for (int i = 0; i < size; ++i) data[i] = .1f * (i % 10) / 128;
}

template <typename T>
bool checkCorrect(T* actual, T* reference, int size) {
  std::cout << std::fixed;
  std::cout.precision(6);
  for (int i = 0; i < size; ++i)
    if (std::abs(actual[i] - reference[i]) >=
        std::numeric_limits<T>::epsilon()) {
      std::cout << "index: " << i << " expected: " << reference[i]
                << " vs actual: " << actual[i] << std::endl;
      return false;
    }
  return true;
}

#endif _GPU_MATMUL_H_
