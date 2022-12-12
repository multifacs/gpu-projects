#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "include/matrix.h"

#define CLOCK std::chrono::high_resolution_clock::now()

int main(void) {
  std::vector<cl_platform_id> platforms;

  cl_uint count = getPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;

  if (count < 1) {
    std::cout << "Platform error\n";
    return 0;
  }

  std::cout << "Finding devices...\n";

  for (size_t i = 0; i < count; i++) {
    cl_platform_id platform = platforms[i];

    cl_device_id gpu = getDevice(1 << 2, platform);
    if (gpu != nullptr) gpus.push_back(std::make_pair(platform, gpu));

    cl_device_id cpu = getDevice(1 << 1, platform);
    if (cpu != nullptr) cpus.push_back(std::make_pair(platform, cpu));
  }

  std::cout << "OK\n\n";

  const int m = 1024;
  const int n = 1024;
  const int k = 1024;

  const size_t a_size = m * n;
  const size_t b_size = n * k;
  const size_t c_size = m * k;
  float* A = new float[a_size];
  float* B = new float[b_size];
  float* C = new float[c_size];
  float* C_reference = new float[c_size];

  createData<float>(A, a_size);
  createData<float>(B, b_size);

  std::cout << " --- MATRIX MULT ---\n\n";

  auto t0 = CLOCK;
  matrix_multiplication(m, n, k, A, B, C);
  auto t1 = CLOCK;
  std::cout << "SEQ - " << TIME_MS(t0, t1) << std::endl;
  std::memcpy(C_reference, C, c_size * sizeof(float));

  t0 = CLOCK;
  matrix_multiplication_omp(m, n, k, A, B, C);
  t1 = CLOCK;
  std::cout << "OMP - " << TIME_MS(t0, t1) << std::endl;
  CHECK_CORRECT(true, float, C_reference, C, c_size)

  timer time;
  matrix_multiplication_gpu(m, n, k, A, B, C, gpus[0], time);
  std::cout << "GPU - " << TIME_MS(time.first, time.second) << std::endl;
  CHECK_CORRECT(true, float, C_reference, C, c_size)

  matrix_multiplication_gpu(m, n, k, A, B, C, cpus[0], time);
  std::cout << "CPU - " << TIME_MS(time.first, time.second) << std::endl;
  CHECK_CORRECT(true, float, C_reference, C, c_size)

  gemm(m, n, k, A, B, C, gpus[0], time);
  std::cout << "GPU OPTIMIZED - " << TIME_MS(time.first, time.second)
            << std::endl;
  CHECK_CORRECT(true, float, C_reference, C, c_size)

  gemm(m, n, k, A, B, C, cpus[0], time);
  std::cout << "CPU OPTIMIZED - " << TIME_MS(time.first, time.second)
            << std::endl;
  CHECK_CORRECT(true, float, C_reference, C, c_size)

  return 0;
}