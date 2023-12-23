#include "utils.cuh"

void check_error(cudaError_t result, const char* const func, const char* const file, const int line) {
    if (result) {
        unsigned int res = static_cast<unsigned int>(result);
        std::cerr << "CUDA error = "<<res<<" at "<<file<<":"<<line<<" '"<<func<<"'\n";
        cudaDeviceReset();
        exit(1);
    }
}

// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
void printDevicesInfos(){
    int nbDevices;
    cudaGetDeviceCount(&nbDevices);
  
    printf("Number of devices: %d\n", nbDevices);
    for (int i=0; i<nbDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d\n", i+1);
        printf("  Device name: %s\n",                       prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",           prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",           prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",    prop.memoryClockRate*2.0*(prop.memoryBusWidth/8.0)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",     (float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n", (float)(prop.sharedMemPerBlock)/1024.0);
        printf("  minor-major: %d-%d\n",                    prop.minor, prop.major);
        printf("  Warp-size: %d\n",                         prop.warpSize);
        printf("  Concurrent kernels: %s\n",                prop.concurrentKernels?"yes":"no");
        printf("  Concurrent computation/communication: %s\n",  prop.deviceOverlap?"yes":"no");
        printf("  Threads per block : %d, Max [X:%d, Y:%d, Z:%d]\n", prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Number of blocks per multiprocessor: %d\n",   prop.maxBlocksPerMultiProcessor);
        printf("  Number of streaming multiprocessors : %d\n",  prop.multiProcessorCount);
        printf("  Number of threads per multiprocessors : %d\n",  prop.maxThreadsPerMultiProcessor);
    }
}