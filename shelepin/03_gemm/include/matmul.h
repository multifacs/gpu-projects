#ifndef _GPU_MATMUL_H_
#define _GPU_MATMUL_H_

// #define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define THREADS 12
#define BLOCK 16

#include <vector>
#include "CL/cl.h"
#include "utils.h"

void matmul(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
void matmul_omp(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c);
void matmul_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

void gemm_cl(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
	std::pair<cl_platform_id, cl_device_id>& dev_pair, timer& time);

std::string ltrim(const std::string& s);
std::string rtrim(const std::string& s);
std::string trim(const std::string& s);

#endif _GPU_MATMUL_H_
