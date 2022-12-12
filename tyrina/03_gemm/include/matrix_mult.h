#ifndef _GPU_MATMUL_H_
#define _GPU_MATMUL_H_

#define THREADS 8
#define BLOCK 16

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CL/cl.h"

void mult_seq(const size_t m, const size_t n, const size_t k,
                           const float* a, const float* b, float* c);
void mult_omp(const size_t m, const size_t n, const size_t k,
                               const float* a, const float* b, float* c);
double mult_gpu(
    const size_t m, const size_t n, const size_t k, const float* a,
    const float* b, float* c,
    std::pair<cl_platform_id, cl_device_id>& dev_pair);

double mult_gemm(const size_t m, const size_t n, const size_t k, const float* a,
            const float* b, float* c,
            std::pair<cl_platform_id, cl_device_id>& dev_pair);

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

template <typename T>
void fillMatrix(T* data, const size_t size) {
  for (int i = 0; i < size; ++i) data[i] = i;
}

template <typename T>
bool compare(T* actual, T* reference, int size) {
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

template <typename T>
std::string check(bool flag, T* result, T* reference, size_t size) {
  if (flag) {
    bool res = compare<T>(result, reference, size);
    if (res) {
      return "PASSED";
    }
    return "FAILED";
  }
}

#endif _GPU_MATMUL_H_
