// saxpy.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "CycleTimer.h"

static float* g_dx = nullptr;
static float* g_dy = nullptr;
static float* g_dres = nullptr;
static int    g_allocN = 0;

__global__ void
saxpy_kernel(int N, float alpha,
             const float* __restrict__ x,
             const float* __restrict__ y,
             float* __restrict__ result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) result[index] = alpha * x[index] + y[index];
}

void saxpyCudaReuseDeviceBuffer(int N, float alpha, float* xarray, float* yarray, float* resultarray) {

    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate device buffers once and reuse
    if (N != g_allocN) {
        if (g_dx) { cudaFree(g_dx); cudaFree(g_dy); cudaFree(g_dres); }
        cudaMalloc(&g_dx,   N * sizeof(float));
        cudaMalloc(&g_dy,   N * sizeof(float));
        cudaMalloc(&g_dres, N * sizeof(float));
        g_allocN = N;
    }

    // Measure end-to-end (H2D + kernel + D2H) with events
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(g_dx, xarray, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_dy, yarray, N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy_kernel<<<blocks, threadsPerBlock>>>(N, alpha, g_dx, g_dy, g_dres);

    cudaMemcpy(resultarray, g_dres, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    int totalBytes = sizeof(float) * 3 * N;
    auto GBPerSec = [](int bytes, float sec){
        return static_cast<float>(bytes) / (1024.f*1024.f*1024.f) / sec;
    };
    printf("End-to-end: %.3f ms\t[%.3f GB/s]\n",
           ms, GBPerSec(totalBytes, ms/1000.0f));

    // Error check
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: CUDA error: code=%d, %s\n",
                errCode, cudaGetErrorString(errCode));
    }
}

// (Optional) add a cleanup you can call once at program end if desired:
void saxpyCleanup() {
    if (g_dx) { cudaFree(g_dx); g_dx=nullptr; }
    if (g_dy) { cudaFree(g_dy); g_dy=nullptr; }
    if (g_dres) { cudaFree(g_dres); g_dres=nullptr; }
    g_allocN = 0;
}

void printCudaInfo() {

    // print out stats about the GPU in the machine.  Useful if
    // students want to know what GPU they are running on.

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
