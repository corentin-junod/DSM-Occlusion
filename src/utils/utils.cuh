#pragma once

#include "definitions.cuh"

#define checkError(val) check_error((val), #val, __FILE__, __LINE__)

void* allocGPU(const uint size, const uint count = 1);
void freeGPU(void* const memory);
void syncGPU();
void check_error(const cudaError_t result, const char* const func, const char* const file, const int line);
void memGpuToCpu(void* const to, const void* const from, const uint size);
void memCpuToGpu(void* const to, const void* const from, const uint size);
void printDevicesInfos();
