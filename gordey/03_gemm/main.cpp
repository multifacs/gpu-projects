#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "include/matrix.h"

#define m 512
#define n 512
#define k 512

int main(int argc, char** argv) {
  std::vector<cl_platform_id> platforms;

  cl_uint count = getPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;

  if (count < 1) {
    throw -1;
  }

  for (size_t i = 0; i < count; i++) {
    cl_platform_id platform = platforms[i];

    cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
    if (gpu != nullptr) gpus.push_back(std::make_pair(platform, gpu));

    cl_device_id cpu = getDevice(CL_DEVICE_TYPE_CPU, platform);
    if (cpu != nullptr) cpus.push_back(std::make_pair(platform, cpu));
  }

  const size_t a_size = m * n;
  const size_t b_size = n * k;
  const size_t c_size = m * k;
  float* a = new float[a_size];
  float* b = new float[b_size];
  float* c = new float[c_size];
  float* c_ref = new float[c_size];

  createData<float>(a, a_size);
  createData<float>(b, b_size);

  auto t0 = std::chrono::high_resolution_clock::now();
  matrix_multiplication(m, n, k, a, b, c);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "SEQ - " << TIME_MS(t0, t1) << std::endl;
  std::memcpy(c_ref, c, c_size * sizeof(float));

  t0 = std::chrono::high_resolution_clock::now();
  matrix_multiplication_omp(m, n, k, a, b, c);
  t1 = std::chrono::high_resolution_clock::now();
  std::cout << "OMP - " << TIME_MS(t0, t1) << std::endl;
  CHECK_CORRECT(true, float, c_ref, c, c_size)

  timer time;
  matrix_multiplication_gpu(m, n, k, a, b, c, gpus[0], time);
  std::cout << "GPU - " << TIME_MS(time.first, time.second) << std::endl;
  CHECK_CORRECT(true, float, c_ref, c, c_size)

  matrix_multiplication_gpu(m, n, k, a, b, c, cpus[0], time);
  std::cout << "CPU - " << TIME_MS(time.first, time.second) << std::endl;
  CHECK_CORRECT(true, float, c_ref, c, c_size)

  gemm(m, n, k, a, b, c, gpus[0], time);
  std::cout << "GPU OPTIMIZED - " << TIME_MS(time.first, time.second)
            << std::endl;
  CHECK_CORRECT(true, float, c_ref, c, c_size)

  gemm(m, n, k, a, b, c, cpus[0], time);
  std::cout << "CPU OPTIMIZED - " << TIME_MS(time.first, time.second)
            << std::endl;
  CHECK_CORRECT(true, float, c_ref, c, c_size)

  return 0;
}