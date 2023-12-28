#pragma once

#include <cstddef>

template<typename T>
class Array{
public:

    __host__ __device__ Array(T* data, std::size_t size): data(data), dataSize(size){};
    __host__ __device__ std::size_t size() const {return dataSize;};

    __host__ __device__ T& operator[](int i) {return data[i];}

    __host__ __device__ T* begin() const {return &data[0];}
    __host__ __device__ T* end()   const {return &data[dataSize];}

private:
    const std::size_t dataSize;
    T* const data;
};

/*
template<typename T>
class MutableArray{
public:

    __host__ __device__ MutableArray(T* data, std::size_t size): data(data), dataSize(size){};
    __host__ __device__ std::size_t size() const {return dataSize;};

    __host__ __device__       T& operator[](int i)       {return data[i];}
    __host__ __device__ const T& operator[](int i) const {return data[i];}

    __host__ __device__  T* begin() const {return &data[0];}
    __host__ __device__  T* end()   const {return &data[dataSize];}

    __host__ __device__  void add(T element) {
        data[dataSize] = element;
        dataSize++;
    }

private:
    std::size_t dataSize;
    T* data;
};
*/