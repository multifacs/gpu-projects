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

  cl_uint platformCount;

  {
    platformCount = 0;
    clGetPlatformIDs(0, nullptr, &platformCount);

    if (platformCount == 0) {
      throw -1;
    }

    cl_platform_id* platformsArray = new cl_platform_id[platformCount];
    clGetPlatformIDs(platformCount, platformsArray, nullptr);
    for (cl_uint i = 0; i < platformCount; ++i) {
      char platformName[128];
      clGetPlatformInfo(platformsArray[i], CL_PLATFORM_NAME, 128, platformName,
                        nullptr);
      cl_uint cpuCount = 0;
      clGetDeviceIDs(platformsArray[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuCount);
      cl_device_id* cpus = new cl_device_id[cpuCount];
      clGetDeviceIDs(platformsArray[i], CL_DEVICE_TYPE_CPU, cpuCount, cpus, nullptr);
      for (cl_uint j = 0; j < cpuCount; ++j) {
        char cpuName[128];
        clGetDeviceInfo(cpus[j], CL_DEVICE_NAME, 128, cpuName, nullptr);
      }
      cl_uint gpuCount = 0;
      clGetDeviceIDs(platformsArray[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
      cl_device_id* gpus = new cl_device_id[gpuCount];
      clGetDeviceIDs(platformsArray[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, nullptr);
      for (cl_uint j = 0; j < gpuCount; ++j) {
        char gpuName[128];
        clGetDeviceInfo(gpus[j], CL_DEVICE_NAME, 128, gpuName, nullptr);
      }
      platforms.push_back(platformsArray[i]);
    }

    delete[] platformsArray;
  }

  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;

  for (size_t i = 0; i < platformCount; i++) {
    cl_platform_id platform = platforms[i];

    cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
    if (gpu != nullptr) gpus.push_back(std::make_pair(platform, gpu));
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