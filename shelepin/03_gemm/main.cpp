#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "include/matmul.h"

#define FLAG_CHECK true

#define M 1024
#define N 1024
#define K 1024

int main(int argc, char** argv) {
  std::vector<cl_platform_id> platforms;
  cl_uint platformCount = getCountAndListOfPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;
  if (platformCount < 1) {
    THROW_EXCEPTION(std::string("PlatfromCount"), std::to_string(platformCount))
  }

  for (size_t i = 0; i < platformCount; i++) {
    cl_platform_id platform = platforms[i];

    // Get GPU count on platform && info
    cl_uint gpuCount = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
    cl_device_id* gpus_pointer = new cl_device_id[gpuCount];
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuCount, gpus_pointer,
                   nullptr);

    // cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
    // if (gpu != nullptr)
    //     gpus.push_back(std::make_pair(platform, gpu));

    if (gpus_pointer != nullptr && i == 0)
      for (int i = 0; i < 2; i++) {
        gpus.push_back(std::make_pair(platform, gpus_pointer[i]));
      }

    if (gpus_pointer != nullptr && i == 1)
      for (int i = 0; i < gpuCount; i++) {
        gpus.push_back(std::make_pair(platform, gpus_pointer[i]));
      }

    cl_device_id cpu = getDevice(CL_DEVICE_TYPE_CPU, platform);
    if (cpu != nullptr) cpus.push_back(std::make_pair(platform, cpu));
  }

  const size_t a_size = M * N;
  const size_t b_size = N * K;
  const size_t c_size = M * K;
  float* a = new float[a_size];
  float* b = new float[b_size];
  float* c = new float[c_size];
  float* c_ref = new float[c_size];

  fillData<float>(a, a_size);
  fillData<float>(b, b_size);

  std::cout << "Task 1 Regular matrix multiplication\n" << std::endl;
  try {
    // SEQ
    auto t0 = std::chrono::high_resolution_clock::now();
    matmul(M, N, K, a, b, c);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential: " << TIME_MS(t0, t1) << std::endl;
    std::memcpy(c_ref, c, c_size * sizeof(float));

    // OMP
    t0 = std::chrono::high_resolution_clock::now();
    matmul_omp(M, N, K, a, b, c);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "OMP: " << TIME_MS(t0, t1) << std::endl;
    CHECK(FLAG_CHECK, float, c_ref, c, c_size)

    // CL GPU
    for (int i = 0; i < gpus.size(); i++) {
      timer time;
      matmul_cl(M, N, K, a, b, c, gpus[i], time);
      char name[128];
      clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
      std::string trimmed_name = trim(std::string(name));
      std::cout << "GPU (" << trimmed_name
                << "): " << TIME_MS(time.first, time.second) << std::endl;
      CHECK(FLAG_CHECK, float, c_ref, c, c_size)
    }

    // CL CPU
    for (int i = 0; i < cpus.size(); i++) {
      timer time;
      matmul_cl(M, N, K, a, b, c, cpus[i], time);
      char name[128];
      clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
      std::string trimmed_name = trim(std::string(name));
      std::cout << "CPU (" << trimmed_name
                << "): " << TIME_MS(time.first, time.second) << std::endl;
      CHECK(FLAG_CHECK, float, c_ref, c, c_size)
    }
  } catch (Exception& exception) {
    std::cout << exception.what() << std::endl;
  }

  std::cout << "\n\nTask 2 GEMM\n" << std::endl;
  try {
    // CL GPU
    for (int i = 0; i < gpus.size(); i++) {
      timer time;
      gemm_cl(M, N, K, a, b, c, gpus[i], time);
      char name[128];
      clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
      std::string trimmed_name = trim(std::string(name));
      std::cout << "GPU (" << trimmed_name
                << "): " << TIME_MS(time.first, time.second) << std::endl;
      CHECK(FLAG_CHECK, float, c_ref, c, c_size)
    }

    // CL CPU
    for (int i = 0; i < cpus.size(); i++) {
      timer time;
      gemm_cl(M, N, K, a, b, c, cpus[i], time);
      char name[128];
      clGetDeviceInfo(cpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
      std::string trimmed_name = trim(std::string(name));
      std::cout << "CPU (" << trimmed_name
                << "): " << TIME_MS(time.first, time.second) << std::endl;
      CHECK(FLAG_CHECK, float, c_ref, c, c_size)
    }
  } catch (Exception& exception) {
    std::cout << exception.what() << std::endl;
  }

  return 0;
}