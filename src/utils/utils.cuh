#pragma once

#define checkError(val) check_error((val), #val, __FILE__, __LINE__)

void* allocGPU(const unsigned int size, const unsigned int count = 1);
void freeGPU(void* const memory);
void syncGPU();
void check_error(cudaError_t result, const char* const func, const char* const file, const int line);
void printDevicesInfos();
