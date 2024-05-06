#pragma once

#include "../utils/utils.cuh"

template<typename T>
class Array2D{
public:
    __host__ __device__ Array2D(const uint width, const uint height, const bool allocArray=true): 
    m_width(width), m_height(height){
        m_data = allocArray ? (T*)malloc(sizeof(T)*width*height) : nullptr; // TODO check for malloc errors
    }

    __host__ __device__ ~Array2D(){ if(m_data!=nullptr) free(m_data); }

    __host__ __device__ uint size()   const {return m_width*m_height;}
    __host__ __device__ uint width()  const {return m_width;}
    __host__ __device__ uint height() const {return m_height;}

    __host__ __device__ T& operator[](const uint i)       {return m_data[i];}
    __host__ __device__ T& operator[](const uint i) const {return m_data[i];}

    __host__ __device__ T& at(const uint x, const uint y) const {return m_data[x+y*m_width];}
    __host__ __device__ T* begin() const {return &m_data[0];}
    __host__ __device__ T* end()   const {return &m_data[size()];}

    __host__ Array2D<T>* toGPU() {
        T* dataOnGPU = (T*) allocGPU(sizeof(T), size());
        memCpuToGpu(dataOnGPU, m_data, sizeof(T)*size());

        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        tmpArray.m_data = dataOnGPU;
        Array2D<T>* newArrayOnGPU = (Array2D<T>*) allocGPU(sizeof(Array2D<T>));
        memCpuToGpu(newArrayOnGPU, &tmpArray, sizeof(Array2D<T>));
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
        return newArrayOnGPU;
    }

    __host__ void fromGPU(Array2D<T>* replica){
        Array2D<T> tmpArray = Array2D<T>(m_width, m_height, false);
        memGpuToCpu(&tmpArray, replica, sizeof(Array2D<T>));
        memGpuToCpu(m_data, tmpArray.m_data, sizeof(T)*size());
        freeGPU(tmpArray.m_data);
        freeGPU(replica);
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
    }

private:
    T* m_data;
    const uint m_width;
    const uint m_height;
};

template<typename T>
class Array3D{
public:
    __host__ __device__ Array3D(const uint width, const uint height, const uint depth, const bool allocArray=true): 
    m_width(width), m_height(height), m_depth(depth){
        m_data = allocArray ? (T*)malloc(sizeof(T)*width*height*depth) : nullptr; // TODO check for malloc errors
    }

    __host__ __device__ ~Array3D(){ if(m_data!=nullptr) free(m_data); }

    __host__ __device__ uint size()   const {return m_width*m_height*m_depth;}
    __host__ __device__ uint width()  const {return m_width;}
    __host__ __device__ uint height() const {return m_height;}
    __host__ __device__ uint depth() const {return m_depth;}

    __host__ __device__ T& operator[](const uint i)       {return m_data[i];}
    __host__ __device__ T& operator[](const uint i) const {return m_data[i];}

    __host__ __device__ T& at(const uint x, const uint y, const uint z) const {return m_data[x+y*m_width+z*(m_width*m_height)];}
    __host__ __device__ void set(const uint x, const uint y, const uint z, const T value) const { m_data[x + y * m_width + z * (m_width * m_height)] = value; }
    __host__ __device__ T* atDepth(const uint depth) const {return &m_data[depth*(m_width*m_height)];}
    __host__ __device__ T* begin() const {return &m_data[0];}
    __host__ __device__ T* end()   const {return &m_data[size()];}

    __host__ Array3D<T>* toGPU() {
        T* dataOnGPU = (T*) allocGPU(sizeof(T), size());
        memCpuToGpu(dataOnGPU, m_data, sizeof(T)*size());

        Array3D<T> tmpArray = Array3D<T>(m_width, m_height, m_depth, false);
        tmpArray.m_data = dataOnGPU;
        Array3D<T>* newArrayOnGPU = (Array3D<T>*) allocGPU(sizeof(Array3D<T>));
        memCpuToGpu(newArrayOnGPU, &tmpArray, sizeof(Array3D<T>));
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
        return newArrayOnGPU;
    }

    __host__ void fromGPU(Array3D<T>* replica){
        Array3D<T> tmpArray = Array3D<T>(m_width, m_height, m_depth, false);
        memGpuToCpu(&tmpArray, replica, sizeof(Array3D<T>));
        memGpuToCpu(m_data, tmpArray.m_data, sizeof(T)*size());
        freeGPU(tmpArray.m_data);
        freeGPU(replica);
        tmpArray.m_data = nullptr; // Avoid freeing a pointer on GPU because tmpArray goes out of scope
    }

private:
    T* m_data;
    const uint m_width;
    const uint m_height;
    const uint m_depth;
};