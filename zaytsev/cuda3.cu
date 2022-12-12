#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <omp.h>

__global__ void float_matrix_multiplication_kernel(const int m, const int n, const int k, float* x, float* y, float* z)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < k && row < m) {
        for (int i = 0; i < n; i++)
            z[row * k + col] += x[row * n + i] * y[i * k + col];
    }
}

__global__ void block_float_matrix_multiplication_kernel(const int m, const int n, const int k, float* x, float* y, float* z)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float res = 0;
    __shared__ float a_block[16 * 16];
    __shared__ float b_block[16 * 16];
    if (col < k && row < m) {
        for (int i = 0; i < n; i += blockDim.y)
        {
            a_block[threadIdx.y * blockDim.x + threadIdx.x] = x[(blockIdx.y * blockDim.y + threadIdx.y) * n + (i + threadIdx.x)];
            b_block[threadIdx.y * blockDim.y + threadIdx.x] = y[(i + threadIdx.y) * k + (blockIdx.x * blockDim.x + threadIdx.x)];
            __syncthreads();
            for (int j = 0; j < blockDim.x; j++) {
                res += a_block[threadIdx.y * blockDim.y + j] * b_block[j * blockDim.y + threadIdx.x];
            }
            __syncthreads();
        }
        z[row * k + col] += res;
    }
}

void float_matrix_multiplication_cuda(const int m, const int n, const int k, const float* x, const float* y, float* z, const dim3 dimGrid, const dim3 dimBlock)
{

    cudaError_t cudaStatus;

    float* gpuX, * gpuY, * gpuZ;
    cudaStatus = cudaMalloc((void**)&gpuX, n * m * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuX) faild\n");
        return;
    }
    cudaStatus = cudaMalloc((void**)&gpuY, n * k * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuY) faild\n");
        return;
    }
    cudaStatus = cudaMalloc((void**)&gpuZ, m * k * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuZ) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuX, x, n * m * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuY, y, n * k * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuZ, z, m * k * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
        return;
    }

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float_matrix_multiplication_kernel << <dimGrid, dimBlock >> > (m, n, k, gpuX, gpuY, gpuZ);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GPU = %f", gpuTime / 1000);
    printf("\n");

    cudaStatus = cudaMemcpy(z, gpuZ, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
        return;
    }

    cudaFree(gpuX); cudaFree(gpuY); cudaFree(gpuZ);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;

}

void block_float_matrix_multiplication_cuda(const int m, const int n, const int k, const float* x, const float* y, float* z, const dim3 dimGrid, const dim3 dimBlock)
{

    cudaError_t cudaStatus;

    float* gpuX, * gpuY, * gpuZ;
    cudaStatus = cudaMalloc((void**)&gpuX, n * m * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuX) faild\n");
        return;
    }
    cudaStatus = cudaMalloc((void**)&gpuY, n * k * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuY) faild\n");
        return;
    }
    cudaStatus = cudaMalloc((void**)&gpuZ, m * k * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc(gpuZ) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuX, x, n * m * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuX) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuY, y, n * k * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuY) faild\n");
        return;
    }

    cudaStatus = cudaMemcpy(gpuZ, z, m * k * sizeof(float),
        cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
        return;
    }

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    block_float_matrix_multiplication_kernel << <dimGrid, dimBlock >> > (m, n, k, gpuX, gpuY, gpuZ);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GEMM GPU time = %f", gpuTime / 1000);
    printf("\n");

    cudaStatus = cudaMemcpy(z, gpuZ, m * k * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy(gpuZ) faild\n");
        return;
    }

    cudaFree(gpuX); cudaFree(gpuY); cudaFree(gpuZ);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return;

}

void float_matrix_multiplication(const int m, const int n, const int k, float* x, float* y, float* z)
{
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                z[i * k + p] += x[i * n + j] * y[j * k + p];
}

void float_matrix_multiplication_omp(const int m, const int n, const int k, float* x, float* y, float* z)
{
#pragma omp parallel for
    for (int i = 0; i < m; ++i)
    {
        float* c = z + i * n;
        for (int j = 0; j < n; ++j)
            c[j] = 0;
        for (int p = 0; p < k; ++p)
        {
            const float* b = y + p * n;
            float a = x[i * k + p];
            for (int j = 0; j < n; ++j)
                c[j] += a * b[j];
        }
    }
}

int main()
{
    const int n_1 = 512;
    const int m_1 = 512;
    const int k_1 = 512;

    float* x_1 = new float[n_1 * m_1];
    float* y_1 = new float[n_1 * k_1];
    float* z_1 = new float[m_1 * k_1];

    for (int i = 0; i < n_1 * m_1; i++)
    {
        x_1[i] = 1.0;
    }
    for (int i = 0; i < n_1 * k_1; i++)
    {
        y_1[i] = 1.0;
    }
    for (int i = 0; i < m_1 * k_1; i++)
    {
        z_1[i] = 0.0;
    }

    float start_1 = omp_get_wtime();

    float_matrix_multiplication(m_1, n_1, k_1, x_1, y_1, z_1);

    float end_1 = omp_get_wtime();

    printf("Seq. = %f", end_1 - start_1);
    printf("\n");

    delete[] x_1, delete[] y_1, delete[] z_1;

    x_1 = new float[n_1 * m_1];
    y_1 = new float[n_1 * k_1];
    z_1 = new float[m_1 * k_1];

    for (int i = 0; i < n_1 * m_1; i++)
    {
        x_1[i] = 1.0;
    }
    for (int i = 0; i < n_1 * k_1; i++)
    {
        y_1[i] = 1.0;
    }
    for (int i = 0; i < m_1 * k_1; i++)
    {
        z_1[i] = 0.0;
    }

    start_1 = omp_get_wtime();

    float_matrix_multiplication_omp(m_1, n_1, k_1, x_1, y_1, z_1);

    end_1 = omp_get_wtime();

    printf("OMP = %f", end_1 - start_1);
    printf("\n");

    delete[] x_1, delete[] y_1, delete[] z_1;

    const int n_0f = 512;
    const int m_0f = 512;
    const int k_0f = 512;

    float* x_0f = new float[n_0f * m_0f];
    float* y_0f = new float[n_0f * k_0f];
    float* z_0f = new float[m_0f * k_0f];

    dim3 dimBlockf(16, 16);
    dim3 dimGridf((m_0f + dimBlockf.x - 1) / dimBlockf.x, (k_0f + dimBlockf.y - 1) / dimBlockf.y);

    for (int i = 0; i < n_0f * m_0f; i++)
    {
        x_0f[i] = 1.0;
    }
    for (int i = 0; i < n_0f * k_0f; i++)
    {
        y_0f[i] = 1.0;
    }
    for (int i = 0; i < m_0f * k_0f; i++)
    {
        z_0f[i] = 0.0;
    }

    float_matrix_multiplication_cuda(m_0f, n_0f, k_0f, x_0f, y_0f, z_0f, dimGridf, dimBlockf);

    delete[] x_0f, delete[] y_0f, delete[] z_0f;


    const int n_1f = 512;
    const int m_1f = 512;
    const int k_1f = 512;

    dim3 dimBlock_1f(16, 16);
    dim3 dimGrid_1f((m_1f + dimBlock_1f.x - 1) / dimBlock_1f.x, (k_1f + dimBlock_1f.y - 1) / dimBlock_1f.y);

    float* x_1f = new float[n_1f * m_1f];
    float* y_1f = new float[n_1f * k_1f];
    float* z_1f = new float[m_1f * k_1f];

    for (int i = 0; i < n_1f * m_1f; i++)
    {
        x_1f[i] = 1.0;
    }
    for (int i = 0; i < n_1f * k_1f; i++)
    {
        y_1f[i] = 1.0;
    }
    for (int i = 0; i < m_1f * k_1f; i++)
    {
        z_1f[i] = 0.0;
    }

    block_float_matrix_multiplication_cuda(m_1f, n_1f, k_1f, x_1f, y_1f, z_1f, dimGrid_1f, dimBlock_1f);

    delete[] x_1f, delete[] y_1f, delete[] z_1f;

    return 0;
}
