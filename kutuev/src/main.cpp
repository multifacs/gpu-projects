#include <CL/cl.h>
#include <stdio.h>

#include <iostream>

const char* kernel_1 =
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

const char* kernel_2 =
    "__kernel void calculate(__global int* a) {    \n"
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

  // Устройство
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  // Контекст
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  {
    // Размерности
    size_t global_size = 20;
    size_t local_size = 5;

    // Создаем элементы opencl
    cl_program program =
        clCreateProgramWithSource(context, 1, &kernel_1, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "information", NULL);
    // Запускаем ядро
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0,
                           NULL, NULL);
    clFlush(queue);
    clFinish(queue);
    // Освобождаем
    clReleaseProgram(program);
    clReleaseKernel(kernel);
  }

  {
    // Исходный массив
    size_t a_size = 8;
    cl_uint* a = (cl_uint*)malloc(a_size * sizeof(cl_uint));
    for (size_t i = 0; i < a_size; i++) {
      a[i] = i;
    }
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }
    printf("\n");

    // Создаем элементы opencl
    cl_program program =
        clCreateProgramWithSource(context, 1, &kernel_2, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "calculate", NULL);
    cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   a_size * sizeof(cl_uint), NULL, NULL);

    // Пишем в буфер
    clEnqueueWriteBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
                         0, NULL, NULL);
    // Запускаем ядро
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memory);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &a_size, NULL, 0, NULL,
                           NULL);
    // Читаем из буфера
    clEnqueueReadBuffer(queue, memory, CL_TRUE, 0, a_size * sizeof(cl_uint), a,
                        0, NULL, NULL);
    clFlush(queue);
    clFinish(queue);
    for (size_t i = 0; i < a_size; i++) {
      printf("%d ", a[i]);
    }

    // Освобождаем
    clReleaseMemObject(memory);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
  }

  // Освобождаем
  clReleaseContext(context);
  clReleaseCommandQueue(queue);

  return 0;
}