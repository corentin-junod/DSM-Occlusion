#pragma once

template<typename T>
class Vec3{

public:
    __host__ __device__ Vec3(T x, T y, T z);

    __host__ __device__ void add(const Vec3* const other);
    __host__ __device__ void sub(const Vec3* const other);
    __host__ __device__ void mul(const Vec3* const other);
    __host__ __device__ void div(const Vec3* const other);

    __host__ __device__ void dot(const Vec3* const other);

    __host__ __device__ Vec3 clone() const ;

private:
    T x;
    T y;
    T z;
};