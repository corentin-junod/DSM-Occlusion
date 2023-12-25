#pragma once

#include <cstddef>

template<typename T>
class SizedArray{
public:

    __host__ __device__ SizedArray(T* data, std::size_t size): data(data), size(size){};
    __host__ __device__ std::size_t getSize() const {return size;};

    __host__ __device__ T  operator[](int i) const {return data[i];}
    __host__ __device__ T& operator[](int i)       {return data[i];}

    __host__ __device__ T* begin(){return &data[0];}
    __host__ __device__ T* end(){return &data[size-1];}

private:
    const std::size_t size;
    T* data;
};