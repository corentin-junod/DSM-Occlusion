#pragma once

#include "../utils/utils.cuh"

template<typename T>
class Array{
public:

    __host__ __device__ Array(T* data, const unsigned int size): data(data), dataSize(size){};
    __host__ __device__ unsigned int size() const {return dataSize;};

    __host__ __device__ T& operator[](int i) {return data[i];}
    __host__ __device__ T& operator[](int i) const {return data[i];}

    __host__ __device__ T* begin() const {return &data[0];}
    __host__ __device__ T* end()   const {return &data[dataSize];}

private:
    T* data;
    unsigned int dataSize;
};

template<typename T>
class Array2D{
public:
    __host__ Array2D(const unsigned int width, const unsigned int height, const bool allocArray=true): m_width(width), m_height(height){
        m_data = allocArray ? (T*)malloc(sizeof(T)*width*height) : nullptr; // TODO check for malloc errors
    }

    __host__ __device__ ~Array2D(){if(m_data != nullptr)free(m_data);}

    __host__ __device__ unsigned int size()   const {return m_width*m_height;}
    __host__ __device__ unsigned int width()  const {return m_width;}
    __host__ __device__ unsigned int height() const {return m_height;}

    __host__ __device__ T& operator[](const unsigned int i)       {return m_data[i];}
    __host__ __device__ T& operator[](const unsigned int i) const {return m_data[i];}

    __host__ __device__ T& at(const unsigned int x, const unsigned int y) const {return m_data[x+y*m_width];}
    __host__ __device__ T* begin() const {return &m_data[0];}
    __host__ __device__ T* end()   const {return &m_data[m_width*m_height];}

    __host__ Array2D<T>* createReplicaGPU() {
        T* dataOnGPU = (T*) allocGPU(m_width*m_height, sizeof(T));
        checkError(cudaMemcpy(dataOnGPU, m_data, sizeof(T)*m_width*m_height, cudaMemcpyHostToDevice));

        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        tmpArray.m_data = dataOnGPU;
        Array2D<T>* newArrayOnGPU = (Array2D<T>*) allocGPU(sizeof(Array2D<T>));
        checkError(cudaMemcpy(newArrayOnGPU, &tmpArray, sizeof(Array2D<T>), cudaMemcpyHostToDevice));
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
        return newArrayOnGPU;
    }

    __host__ void consumeReplicaGPU(Array2D<T>* replica){
        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        checkError(cudaMemcpy(&tmpArray, replica, sizeof(Array2D<T>), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(m_data, tmpArray.m_data, sizeof(T)*m_width*m_height, cudaMemcpyDeviceToHost));
        freeGPU(tmpArray.m_data);
        freeGPU(replica);
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
    }

private:
    T* m_data;
    const unsigned int m_width;
    const unsigned int m_height;
};