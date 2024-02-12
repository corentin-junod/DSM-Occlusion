#pragma once

#include "definitions.cuh"
#define checkError(val) check_error((val), #val, __FILE__, __LINE__)

void* allocGPU(const uint size, const uint count = 1);
void freeGPU(void* const memory);
void syncGPU();
void check_error(cudaError_t result, const char* const func, const char* const file, const int line);
void memGpuToCpu(void* to, const void* from, const uint size);
void memCpuToGpu(void* to, const void* from, const uint size);
void printDevicesInfos();
