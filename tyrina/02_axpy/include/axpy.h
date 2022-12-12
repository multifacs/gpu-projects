#ifndef _LAB02_AXPY_
#define _LAB02_AXPY_

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#define THREADS omp_get_max_threads()

#define TIME_US(t0, t1) std::chrono::duration_cast<us>(t1 - t0).count() << " us"
#define TIME_MS(t0, t1) std::chrono::duration_cast<ms>(t1 - t0).count() << " ms"
#define TIME_S(t0, t1) std::chrono::duration_cast<s>(t1 - t0).count() << " s"

using us = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using s = std::chrono::seconds;
using timer = std::pair<std::chrono::high_resolution_clock::time_point,
                        std::chrono::high_resolution_clock::time_point>;

cl_program createProgramFromSource(cl_context ctx, const char* file);
cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl);
cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id);

size_t getTheClosestBiggerDegreeOf2(const size_t x);

void saxpy(const int& n, const float a, const float* x, const int& incx,
           float* y, const int& incy);
void daxpy(const int& n, const double a, const double* x, const int& incx,
           double* y, const int& incy);

void saxpy_omp(const int& n, const float a, const float* x, const int& incx,
               float* y, const int& incy);
void daxpy_omp(const int& n, const double a, const double* x, const int& incx,
               double* y, const int& incy);

void saxpy_cl(int n, float a, const float* x, int incx, float* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group_size = 256);
void daxpy_cl(int n, double a, const double* x, int incx, double* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group_size = 256);

template <typename T>
void fillData(T* data, const size_t size) {
  for (int i = 0; i < size; ++i) data[i] = .1f * (i % 10) / 128;
}

// ------------------------------------------------------------------------------------
// Functions for check for accuracy
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

template <typename T>
std::string check(bool flag, T* result, T* reference, size_t size) {
  if (flag) {
    bool res = checkCorrect<T>(result, reference, size);
    if (res) {
      return "PASSED";
    }
    return "FAILED";
  }
}

#endif  // _LAB02_AXPY_
