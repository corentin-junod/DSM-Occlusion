#pragma once

#include "Vec3.cuh"
#include "Point3.cuh"

#include <iostream>

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

    template<typename U> __host__ friend std::ostream& operator<<(std::ostream& os, const Ray<U>& ray);

private:
    Point3<T>& origin;
    Vec3<T>& direction;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Ray<T>& ray){
    os << "Ray: Origin("<<ray.origin.x<<","<<ray.origin.y<<","<<ray.origin.z<<") Direction("<<ray.direction.x<<","<<ray.direction.y<<","<<ray.direction.z<<")";
    return os;
};