#pragma once

template<typename T>
class Point3{
public:
    __host__ __device__ Point3(T x, T y, T z) : x(x), y(y), z(z){};
    T x;
    T y;
    T z;
};