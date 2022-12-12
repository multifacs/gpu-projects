#include <omp.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "include/matrix_mult.h"

#define m 1024
#define n 1024
#define k 1024

int main(int argc, char** argv) {
  std::vector<cl_platform_id> platforms;

  cl_uint count = getPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;

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

  fillMatrix<float>(a, a_size);
  fillMatrix<float>(b, b_size);

  auto t0 = omp_get_wtime();
  mult_seq(m, n, k, a, b, c);
  auto t1 = omp_get_wtime();
  std::cout << "SEQ: " << t1 - t0 << std::endl;
  std::memcpy(c_ref, c, c_size * sizeof(float));

  t0 = omp_get_wtime();
  mult_omp(m, n, k, a, b, c);
  t1 = omp_get_wtime();
  std::cout << "OMP: " << t1 - t0 << " " << check<float>(true, c_ref, c, c_size)
            << std::endl;

  t0 = mult_gpu(m, n, k, a, b, c, gpus[0]);
  std::cout << "GPU: " << t0 << " " << check<float>(true, c_ref, c, c_size)
            << std::endl;

  t0 = mult_gemm(m, n, k, a, b, c, gpus[0]);
  std::cout << "GEMM: " << t0 << " " << check<float>(true, c_ref, c, c_size)
            << std::endl;

  return 0;
}