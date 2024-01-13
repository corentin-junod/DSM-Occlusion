#pragma once

#include <stdio.h>
#include <iostream>

#define checkError(val) check_error((val), #val, __FILE__, __LINE__)

void* allocMemory(const unsigned int count, const unsigned int size, const bool useGPU);
void* allocGPU(unsigned int count, unsigned int size);
void freeMemory(void* const memory, const bool useGPU);
void freeGPU(void* memory);
void check_error(cudaError_t result, const char* const func, const char* const file, const int line);
void printDevicesInfos();
