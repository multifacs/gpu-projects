#include <CL/cl.h>
#include <stdio.h>

#include <iostream>

const char* source1 =
    "__kernel void information() {                                             "
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

const char* source2 =
    "__kernel void calculate(__global int* a) {    \n"
    "   int id = get_global_id(0);           \n"
    "   a[id] = a[id] + id;      \n"
    "}                                             \n";

int main() {

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

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  {
    size_t global = 15;
    size_t local = 3;
    cl_program program =
        clCreateProgramWithSource(context, 1, &source1, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "information", NULL);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
                           NULL, NULL);
    clFlush(queue);
    clFinish(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
  }

  {
    size_t a_size = 12;
    cl_uint* a = (cl_uint*)malloc(a_size * sizeof(cl_uint));
    for (size_t i = 0; i < a_size; i++) {
      a[i] = i;
    }
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }
    printf("\n");
    cl_program program =
        clCreateProgramWithSource(context, 1, &source2, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculate", NULL);
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