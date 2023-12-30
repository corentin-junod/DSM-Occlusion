#pragma once

template<typename T>
class Point3{
public:
    __host__ __device__ Point3(T x, T y, T z) : x(x), y(y), z(z){};
    __host__ __device__ void set(T newX, T newY, T newZ){x=newX; y=newY; z=newZ;};
    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Point3<T>& p) {
    os << "Point3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}