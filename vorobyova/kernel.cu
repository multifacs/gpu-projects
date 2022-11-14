
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>

__global__ void kernel() {
	printf("Hello, world!\n");
}

__global__ void print_kernel() {
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("I am from %i block, %i thread (global index: %i)\n", blockIdx.x, threadIdx.x, global_index);
}

__global__ void add_kernel(int* a, int n) {
	int ThreadGlobalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int i = ThreadGlobalIndex;
	if (i < n) {
		a[i] = a[i] + ThreadGlobalIndex;
	}
}

int main()
{
	std::cout << " Task 1, 2:" << std::endl;
	kernel <<<2, 2 >>> ();
	print_kernel <<<2, 2 >>> ();
	cudaDeviceSynchronize();

	int n = 100;
	int* a = new int[n];
	int* a_gpu;

	cudaMalloc((void**)& a_gpu, n * sizeof(int));

	for (int i = 0; i < n; i++) {
		a[i] = 0;
	}
	std::cout << "\n";
	std::cout << " Task 3:" << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << a[i] << " ";
	}
	std::cout << "\n";
	cudaMemcpy(a_gpu, a, n * sizeof(int),cudaMemcpyHostToDevice);

	const int block_size = 256;
	int num_blocks = (n + block_size - 1) / block_size;

	add_kernel <<<num_blocks, block_size >>> (a_gpu, n);
	cudaMemcpy(a, a_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << a[i] << " ";
	}
	std::cout << "\n";

	delete[] a;
	cudaFree(a_gpu);

	return 0;
}
