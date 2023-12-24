#pragma once

#include "Vec3.cuh"
#include "Point3.cuh"

template<typename T>
class Ray{

public:
    __host__ __device__ Ray(Point3<T>* origin, Vec3<T>* direction);

    __host__ __device__ static Ray randomInHemisphere();

    __host__ __device__ Vec3<T>* getDirection(){return direction;};
    __host__ __device__ Vec3<T>* getOrigin(){return origin;};

private:
    Vec3<T>* origin;
    Vec3<T>* direction;
};