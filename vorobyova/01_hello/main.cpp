#include <CL/cl.h>
#include <stdio.h>

#include <iostream>

const char* kernel_1 =
    "__kernel void task1() {                                             "
    "                       \n"
    "    int groupId = get_group_id(0);                                        "
    "                       \n"
    "    int localId = get_local_id(0);                                        "
    "                       \n"
    "    int globalId = get_global_id(0);                                      "
    "                       \n"
    "    printf(\"I am from %d block, %d thread (global index: %d)\", groupId, "
    "localId, globalId);    \n"
    "}                                                                         "
    "                       \n";

const char* kernel_2 =
    "__kernel void task2(__global int* a) {    \n"
    "   int globalId = get_global_id(0);           \n"
    "   a[globalId] = a[globalId] + globalId;      \n"
    "}                                             \n";

int main() {
  // Платформа
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
  cl_uint platformCount = 0;
  clGetPlatformIDs(0, NULL, &platformCount);

  for (cl_uint i = 0; i < platformCount; i++) {
    constexpr size_t maxLength = 128;
    char platformName[maxLength];
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, maxLength, platformName,
                      NULL);
    printf("%s\n", platformName);
  }

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  // Контекст и очередь
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, 0, NULL);

  {
    cl_program program =
        clCreateProgramWithSource(context, 1, &kernel_1, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "task1", NULL);
    // Размеры
    size_t globalSize = 25;
    size_t localSize = 5;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0,
                           NULL, NULL);
    clFlush(queue);
    clFinish(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
  }

  {
    size_t a_size = 8;
    cl_uint* a = new cl_uint[a_size];
    for (size_t i = 0; i < a_size; i++) {
      a[i] = 0;
    }
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }
    printf("\n");
    cl_program program =
        clCreateProgramWithSource(context, 1, &kernel_2, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "task2", NULL);
    cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   a_size * sizeof(cl_uint), NULL, NULL);
    clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
                         0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memory);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &a_size, NULL, 0, NULL,
                           NULL);
    clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
                        0, NULL, NULL);
    clFlush(queue);
    clFinish(queue);
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }
    clReleaseMemObject(memory);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
  }

  clReleaseContext(context);
  clReleaseCommandQueue(queue);

  return 0;
}