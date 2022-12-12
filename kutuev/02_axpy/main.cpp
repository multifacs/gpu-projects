#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "include/axpy.h"

#define FLAG_CHECK true

int main(int argc, char** argv) {
  std::vector<cl_platform_id> platforms;
  cl_uint platformCount = getCountAndListOfPlatforms(platforms);
  std::vector<std::pair<cl_platform_id, cl_device_id>> gpus, cpus;
  if (platformCount < 1) {
    return -1;
  }

  for (size_t i = 0; i < platformCount; i++) {
    cl_platform_id platform = platforms[i];

    cl_device_id gpu = getDevice(CL_DEVICE_TYPE_GPU, platform);
    if (gpu != nullptr) gpus.push_back(std::make_pair(platform, gpu));

    cl_device_id cpu = getDevice(CL_DEVICE_TYPE_CPU, platform);
    if (cpu != nullptr) cpus.push_back(std::make_pair(platform, cpu));
  }

  const int n = 150'000'000;
  const int inc_x = 1;
  const int inc_y = 1;

  const int x_size = n * inc_x;
  const int y_size = n * inc_y;

  std::cout << "saxpy" << std::endl << std::endl;
  {
    float a = 10.0;
    float* x = new float[x_size];
    float* y = new float[y_size];
    float* ref = new float[y_size];
    fillData<float>(x, x_size);

    // SEQ
    fillData<float>(y, y_size);
    auto t0 = std::chrono::high_resolution_clock::now();
    saxpy(n, a, x, inc_x, y, inc_y);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "SEQ: " << TIME_MS(t0, t1) << std::endl;
    std::memcpy(ref, y, y_size * sizeof(float));

    // OMP
    fillData<float>(y, y_size);
    t0 = std::chrono::high_resolution_clock::now();
    saxpy_omp(n, a, x, inc_x, y, inc_y);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "OMP: " << TIME_MS(t0, t1) << " "
              << check<float>(FLAG_CHECK, ref, y, y_size) << std::endl;

    std::cout << "GPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      // GPU OPENCL
      for (int i = 0; i < gpus.size(); i++) {
        fillData<float>(y, y_size);
        timer time;
        saxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], time, group_size);
        char name[128];
        clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Size: " << group_size
                  << " GPU: " << TIME_MS(time.first, time.second) << " "
                  << check<float>(FLAG_CHECK, ref, y, y_size) << std::endl;
      }
    }

    delete[] x;
    delete[] y;
    delete[] ref;
  }

  std::cout << std::endl << "daxpy" << std::endl << std::endl;
  {
    double a = 10.0;
    double* x = new double[x_size];
    double* y = new double[y_size];
    double* ref = new double[y_size];
    fillData<double>(x, x_size);

    // SEQ
    fillData<double>(y, y_size);
    auto t0 = std::chrono::high_resolution_clock::now();
    daxpy(n, a, x, inc_x, y, inc_y);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "SEQ: " << TIME_MS(t0, t1) << std::endl;
    std::memcpy(ref, y, y_size * sizeof(double));

    fillData<double>(y, y_size);
    t0 = std::chrono::high_resolution_clock::now();
    daxpy_omp(n, a, x, inc_x, y, inc_y);
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "OMP: " << TIME_MS(t0, t1) << " "
              << check<double>(FLAG_CHECK, ref, y, y_size) << std::endl;

    std::cout << "GPU"
              << "\n";
    for (size_t group_size = 8; group_size <= 256; group_size *= 2) {
      // GPU OPENCL
      for (int i = 0; i < gpus.size(); i++) {
        fillData<double>(y, y_size);
        timer time;
        daxpy_cl(n, a, x, inc_x, y, inc_y, gpus[i], time);
        char name[128];
        clGetDeviceInfo(gpus[i].second, CL_DEVICE_NAME, 128, name, nullptr);
        std::cout << "Size: " << group_size
                  << " GPU: " << TIME_MS(time.first, time.second) << " "
                  << check<double>(FLAG_CHECK, ref, y, y_size) << std::endl;
      }
    }

    delete[] x;
    delete[] y;
    delete[] ref;
  }

  return 0;
}
