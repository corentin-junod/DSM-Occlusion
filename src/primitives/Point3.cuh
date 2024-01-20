#pragma once

#include "Vec3.cuh"

template<typename T>
class Point3{
public:
    __host__ __device__ Point3() : x(0), y(0), z(0){};
    __host__ __device__ Point3(T x, T y, T z) : x(x), y(y), z(z){};

    __host__ __device__ void set(T newX, T newY, T newZ){x=newX; y=newY; z=newZ;};

    __host__ __device__ bool operator==(const Point3<T>& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    __host__ __device__ bool operator!=(const Point3<T>& other) const {
        return x != other.x || y != other.y || z != other.z;
    }

    __host__ __device__ Vec3<T> operator-(const Point3<T>& other) const {
        return Vec3<T>(x-other.x, y-other.y, z-other.z); 
    }

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Point3<T>& p) {
    os << "Point3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}