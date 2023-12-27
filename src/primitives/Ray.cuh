#pragma once

#include "Vec3.cuh"
#include "Point3.cuh"

template<typename T>
class Ray{

public:
    __host__ __device__ Ray(Point3<T>& origin, Vec3<T>& direction) : origin(origin), direction(direction){};
    
    __host__ __device__ Vec3<T>&   getDirection()             const {return direction;};
    __host__ __device__ void       setDirection(Vec3<T>& dir)       {direction=dir;};
    __host__ __device__ void       setDirection(Vec3<T> dir)        {direction=dir;};

    __host__ __device__ Point3<T>& getOrigin()                const {return origin;};
    __host__ __device__ void       setOrigin(Point3<T>& ori)        {origin=ori;};
    __host__ __device__ void       setOrigin(const Point3<T>& ori)  {origin=ori;};

private:
    Point3<T>& origin;
    Vec3<T>& direction;
};