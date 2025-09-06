#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cuda_runtime.h>

void saxpyCuda(int N, float alpha, float* x, float* y, float* result);             // baseline
void saxpyCudaReuseDeviceBuffer(int N, float alpha, float* x, float* y, float* result); // persistent
void printCudaInfo();
void saxpyCleanup(); // no-op in baseline

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv)
{
    int N = 100 * 1000 * 1000;

    int opt;
    static struct option long_options[] = {
        {"arraysize",  1, 0, 'n'},
        {"help",       0, 0, '?'},
        {0 ,0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'n': N = atoi(optarg); break;
        case '?':
        default: usage(argv[0]); return 1;
        }
    }

    const float alpha = 2.0f;
    // float* xarray = new float[N];
    // float* yarray = new float[N];
    // float* resultarray = new float[N];

    // Allocate PINNED host buffers (replaces: new float[N])
    float *xarray, *yarray, *resultarray;
    cudaMallocHost(&xarray,     N * sizeof(float));
    cudaMallocHost(&yarray,     N * sizeof(float));
    cudaMallocHost(&resultarray, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        xarray[i] = yarray[i] = i % 10;
        resultarray[i] = 0.f;
    }

    printCudaInfo();

    printf("Running 3 timing tests:\n");
    for (int i = 0; i < 3; i++) {
    #if defined(BUILD_PERSISTENT)
        saxpyCudaReuseDeviceBuffer(N, alpha, xarray, yarray, resultarray);
    #else
        saxpyCuda(N, alpha, xarray, yarray, resultarray);
    #endif
    }

    // delete [] xarray;
    // delete [] yarray;
    // delete [] resultarray;

    // Free PINNED host buffers (replaces: delete[])
    cudaFreeHost(xarray);
    cudaFreeHost(yarray);
    cudaFreeHost(resultarray);

    #if defined(BUILD_PERSISTENT)
        saxpyCleanup();
    #endif
    
    return 0;
}
