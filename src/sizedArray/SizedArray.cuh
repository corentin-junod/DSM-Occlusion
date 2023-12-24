#pragma once

#include <cstddef>

template<typename T>
class SizedArray{
public:

    __host__ __device__ SizedArray(T* data, std::size_t size): data(data), size(size){};
    __host__ __device__ std::size_t getSize() const {return size;};

    __host__ __device__ T  operator[](int index) const {return data[index];}
    //__host__ __device__ T& operator[](int index)       {return data[index];}

private:
    const std::size_t size;
    const T* const data;
};