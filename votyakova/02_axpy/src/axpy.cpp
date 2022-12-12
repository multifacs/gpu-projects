#include "../include/axpy.h"

void saxpy(const int& n, const float a, const float* x, const int& incx,
           float* y, const int& incy) {
  for (int i = 0; i < n; i++) y[i * incy] += a * x[i * incx];
}
void daxpy(const int& n, const double a, const double* x, const int& incx,
           double* y, const int& incy) {
  for (int i = 0; i < n; i++) y[i * incy] += a * x[i * incx];
}

void saxpy_omp(const int& n, const float a, const float* x, const int& incx,
               float* y, const int& incy) {
#pragma omp parallel for num_threads(THREADS)
  for (int i = 0; i < n; i++) y[i * incy] += a * x[i * incx];
}

void daxpy_omp(const int& n, const double a, const double* x, const int& incx,
               double* y, const int& incy) {
#pragma omp parallel for num_threads(THREADS)
  for (int i = 0; i < n; i++) y[i * incy] += a * x[i * incx];
}

void saxpy_cl(int n, float a, const float* x, int incx, float* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group) {
  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0};

  cl_context context =
      clCreateContext(properties, 1, &dev_pair.second, NULL, NULL, NULL);

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, dev_pair.second, 0, NULL);

  cl_program program =
      createProgramFromSource(context, "kernels/saxpy_kernel.cl");
  clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "saxpy", NULL);

  cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float) * incy * n, NULL, NULL);
  cl_mem x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(float) * incx * n, NULL, NULL);

  clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * incy * n, y,
                       0, NULL, NULL);
  clEnqueueWriteBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(float) * incx * n, x,
                       0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(float), &a);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), &incx);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_buffer);
  clSetKernelArg(kernel, 5, sizeof(int), &incy);

  size_t size = (n % group == 0) ? n : n + group - n % group;

  time.first = std::chrono::high_resolution_clock::now();
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, &group, 0, NULL, NULL);
  clFinish(queue);
  time.second = std::chrono::high_resolution_clock::now();

  clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(float) * incy * n, y,
                      0, NULL, NULL);

  clReleaseMemObject(y_buffer);
  clReleaseMemObject(x_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void daxpy_cl(int n, double a, const double* x, int incx, double* y, int incy,
              std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time,
              size_t group) {
  cl_context_properties properties[3] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)dev_pair.first, 0};

  cl_context context =
      clCreateContext(properties, 1, &dev_pair.second, NULL, NULL, NULL);

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, dev_pair.second, 0, NULL);

  cl_program program =
      createProgramFromSource(context, "kernels/daxpy_kernel.cl");
  clBuildProgram(program, 1, &dev_pair.second, nullptr, nullptr, nullptr);

  cl_kernel kernel = clCreateKernel(program, "daxpy", NULL);

  cl_mem y_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(double) * incy * n, NULL, NULL);
  cl_mem x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                   sizeof(double) * incx * n, NULL, NULL);

  clEnqueueWriteBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(double) * incy * n,
                       y, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, x_buffer, CL_TRUE, 0, sizeof(double) * incx * n,
                       x, 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(double), &a);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &x_buffer);
  clSetKernelArg(kernel, 3, sizeof(int), &incx);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &y_buffer);
  clSetKernelArg(kernel, 5, sizeof(int), &incy);

  size_t size = (n % group == 0) ? n : n + group - n % group;

  time.first = std::chrono::high_resolution_clock::now();
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, &group, 0, NULL, NULL);
  clFinish(queue);
  time.second = std::chrono::high_resolution_clock::now();

  clEnqueueReadBuffer(queue, y_buffer, CL_TRUE, 0, sizeof(double) * incy * n, y,
                      0, NULL, NULL);

  clReleaseMemObject(y_buffer);
  clReleaseMemObject(x_buffer);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

cl_program createProgramFromSource(cl_context ctx, const char* file) {
  std::fstream kernel_file(file, std::ios::in);
  std::string kernel_code((std::istreambuf_iterator<char>(kernel_file)),
                          std::istreambuf_iterator<char>());
  kernel_file.close();
  const char* kernel_code_p = kernel_code.c_str();
  size_t kernel_code_len = kernel_code.size();

  cl_int errorcode = CL_SUCCESS;
  cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_code_p,
                                                 &kernel_code_len, &errorcode);

  return program;
}

cl_device_id getDevice(cl_device_type type, cl_platform_id& plfrm_id) {
  cl_uint device_count = 0;
  clGetDeviceIDs(plfrm_id, type, 0, nullptr, &device_count);

  if (device_count == 0) return nullptr;

  if (device_count > 0) {
    std::vector<cl_device_id> device_vec(device_count);
    clGetDeviceIDs(plfrm_id, type, device_count, device_vec.data(), nullptr);

    if (device_vec.size() > 0) {
      cl_device_id id = device_vec.front();
      return id;
    }
  }
  return nullptr;
}

cl_uint getCountAndListOfPlatforms(std::vector<cl_platform_id>& pl) {
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount);

  if (platformCount == 0) {
    return -1;
  }

  cl_platform_id* platforms = new cl_platform_id[platformCount];
  clGetPlatformIDs(platformCount, platforms, nullptr);
  for (cl_uint i = 0; i < platformCount; ++i) {
    // Get platform info
    char platformName[128];
    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName,
                      nullptr);
    cl_uint cpuCount = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuCount);
    cl_device_id* cpus = new cl_device_id[cpuCount];
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, cpuCount, cpus, nullptr);
    for (cl_uint j = 0; j < cpuCount; ++j) {
      char cpuName[128];
      clGetDeviceInfo(cpus[j], CL_DEVICE_NAME, 128, cpuName, nullptr);
    }

    // Get GPU count on platform && info
    cl_uint gpuCount = 0;
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
    cl_device_id* gpus = new cl_device_id[gpuCount];
    clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, nullptr);
    for (cl_uint j = 0; j < gpuCount; ++j) {
      char gpuName[128];
      clGetDeviceInfo(gpus[j], CL_DEVICE_NAME, 128, gpuName, nullptr);
    }
    pl.push_back(platforms[i]);
  }

  delete[] platforms;
  return platformCount;
}