#pragma once

#include "../utils/utils.cuh"

template<typename T>
class Array2D{
public:
    __host__ __device__ Array2D(const uint width, const uint height, const bool allocArray=true): m_width(width), m_height(height){
        m_data = allocArray ? (T*)malloc(sizeof(T)*width*height) : nullptr; // TODO check for malloc errors
    }

    __host__ __device__ ~Array2D(){if(m_data != nullptr)free(m_data);}

    __host__ __device__ uint size()   const {return m_width*m_height;}
    __host__ __device__ uint width()  const {return m_width;}
    __host__ __device__ uint height() const {return m_height;}

    __host__ __device__ T& operator[](const uint i)       {return m_data[i];}
    __host__ __device__ T& operator[](const uint i) const {return m_data[i];}

    __host__ __device__ T& at(const uint x, const uint y) const {return m_data[x+y*m_width];}
    __host__ __device__ T* begin() const {return &m_data[0];}
    __host__ __device__ T* end()   const {return &m_data[m_width*m_height];}

    __host__ Array2D<T>* toGPU() {
        T* dataOnGPU = (T*) allocGPU(sizeof(T), m_width*m_height);
        checkError(cudaMemcpy(dataOnGPU, m_data, sizeof(T)*m_width*m_height, cudaMemcpyHostToDevice));

        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        tmpArray.m_data = dataOnGPU;
        Array2D<T>* newArrayOnGPU = (Array2D<T>*) allocGPU(sizeof(Array2D<T>));
        checkError(cudaMemcpy(newArrayOnGPU, &tmpArray, sizeof(Array2D<T>), cudaMemcpyHostToDevice));
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
        return newArrayOnGPU;
    }

    __host__ void fromGPU(Array2D<T>* replica){
        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        checkError(cudaMemcpy(&tmpArray, replica, sizeof(Array2D<T>), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(m_data, tmpArray.m_data, sizeof(T)*m_width*m_height, cudaMemcpyDeviceToHost));
        freeGPU(tmpArray.m_data);
        freeGPU(replica);
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
    }

private:
    T* m_data;
    const uint m_width;
    const uint m_height;
};