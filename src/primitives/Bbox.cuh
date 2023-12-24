#pragma once

#include "vector"
#include "limits"

#include "Point3.cuh"
#include "Ray.cuh"

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox(int minX, int maxX, int minY, int maxY, int minZ, int maxZ);
    __host__ __device__ static Bbox getEnglobing(std::vector<Point3<T>*> objList);

    __host__ __device__ Bbox merge(const Bbox* const other) const ;
    __host__ __device__ int getEdgeLength(const int axis) const ;
    __host__ __device__ bool intersects(const Ray<T>& ray);


private:
    const int minX;
    const int maxX;
    const int minY;
    const int maxY;
    const int minZ;
    const int maxZ;
};