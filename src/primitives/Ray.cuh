#pragma once

#include "Vec3.cuh"
#include "Point3.cuh"

template<typename T>
class Ray{

public:
    __host__ __device__ Ray(Point3<T>& origin, Vec3<T>& direction) : origin(origin), direction(direction){};
    __host__ __device__ Vec3<T>& getDirection() const {return direction;}
    __host__ __device__ Point3<T>& getOrigin() const {return origin;};

private:
    Point3<T>& origin;
    Vec3<T>& direction;
};