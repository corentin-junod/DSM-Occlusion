#pragma once

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