#include "utils.cuh"

#include <stdio.h>
#include <iostream>

void* allocGPU(const uint count, const uint size){
    void* result = nullptr;
    checkError(cudaMalloc(&result, ((size_t)count)*((size_t)size)));
    return result;
}

void freeGPU(void* const memory){
    checkError(cudaFree(memory));
}

void syncGPU(){
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());
}

void check_error(const cudaError_t result, const char* const func, const char* const file, const int line) {
    if (result) {
        uint res = static_cast<uint>(result);
        std::cerr << "CUDA error = "<<res<<" ("<< cudaGetErrorString(result) <<") at "<<file<<":"<<line<<" '"<<func<<"'\n";
        cudaDeviceReset();
        exit(1);
    }
}

void memGpuToCpu(void* const to, const void* const from, const size_t size){
    checkError(cudaMemcpy(to, from, size, cudaMemcpyDeviceToHost));
}

void memCpuToGpu(void* const to, const void* const from, const size_t size){
    checkError(cudaMemcpy(to, from, size, cudaMemcpyHostToDevice));
}

// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
void printDevicesInfos(){
    int nbDevices;
    cudaGetDeviceCount(&nbDevices);
  
    printf("***** GPU Information *****\n");
    printf("Number of devices: %d\n", nbDevices);
    for (int i=0; i<nbDevices; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        printf("Device %d\n", i+1);
        printf("  Device name: %s\n",                                p.name);
        printf("  Memory Clock Rate (MHz): %d\n",                    p.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",                    p.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",             p.memoryClockRate*2.0*(p.memoryBusWidth/8.0)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",              (float)(p.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",          (float)(p.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n",                             p.minor, p.major);
        printf("  Warp-size: %d\n",                                  p.warpSize);
        printf("  Concurrent kernels: %s\n",                         p.concurrentKernels?"yes":"no");
        printf("  Concurrent computation/communication: %s\n",       p.deviceOverlap?"yes":"no");
        printf("  Threads per block : %d, Max [X:%d, Y:%d, Z:%d]\n", p.maxThreadsPerBlock, p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  Number of blocks per multiprocessor: %d\n",        p.maxBlocksPerMultiProcessor);
        printf("  Number of streaming multiprocessors : %d\n",       p.multiProcessorCount);
        printf("  Number of threads per multiprocessors : %d\n",     p.maxThreadsPerMultiProcessor);
    }
    printf("**************************\n");
}