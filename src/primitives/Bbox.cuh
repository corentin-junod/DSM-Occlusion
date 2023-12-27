#pragma once

#include "Point3.cuh"
#include "Ray.cuh"
#include "../array/Array.cuh"

#include <limits>

constexpr int MAX_INT = std::numeric_limits<int>::max();
constexpr int MIN_INT = std::numeric_limits<int>::min();

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox(T minX, T maxX, T minY, T maxY, T minZ, T maxZ) :
        minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ), 
        center(Point3<T>((maxX+minX)/2, (maxY+minY)/2, (maxZ+minZ)/2)){};

    __host__ __device__ static Bbox getEnglobing(const Array<Point3<T>>& objList){
        T minX = MAX_INT, minY = MAX_INT, minZ = MAX_INT;
        T maxX = MIN_INT, maxY = MIN_INT, maxZ = MIN_INT;
        for(const Point3<T>& point : objList){
            if(point.x < minX) minX = point.x;
            if(point.x > maxX) maxX = point.x;
            if(point.y < minY) minY = point.y;
            if(point.y > maxY) maxY = point.y;
            if(point.z < minZ) minZ = point.z;
            if(point.z > maxZ) maxZ = point.z;
        }
        return Bbox(minX, maxX, minY, maxY, minZ, maxZ);
    }

    __host__ __device__ T getEdgeLength(const char axis) const {
        switch (axis) {
            case 'X' : return maxX - minX;
            case 'Y' : return maxY - minY;
            default  : return maxZ - minZ;
        }
    }

    __host__ __device__ const Point3<T>& getCenter() const {return center;}

    __host__ __device__ bool intersects(const Ray<T>& ray) const {
        float min, max;

        const Vec3<T>& rayDir = ray.getDirection();
        const Point3<T>& rayOrigin = ray.getOrigin();

        const float xInverse = 1 / rayDir.x;
        const float tNearX = (minX - rayOrigin.x) * xInverse;
        const float tFarX  = (maxX - rayOrigin.x) * xInverse;

        if(tNearX > tFarX){
            min = tFarX;
            max = tNearX;
        }else{
            min = tNearX;
            max = tFarX;
        }

        const float yInverse = 1 / rayDir.y;
        const float tNearY = (minY - rayOrigin.y) * yInverse;
        const float tFarY  = (maxY - rayOrigin.y) * yInverse;

        if(tNearY > tFarY){
            min = min < tFarY  ? tFarY  : min;
            max = max > tNearY ? tNearY : max;
        }else{
            min = min < tNearY ? tNearY : min;
            max = max > tFarY  ? tFarY  : max;
        }

        if(max < min) return false;

        const float zInverse = 1 / rayDir.z;
        const float tNearZ = (minZ - rayOrigin.z) * zInverse;
        const float tFarZ  = (maxZ - rayOrigin.z) * zInverse;

        if(tNearZ > tFarZ){
            min = min < tFarZ  ? tFarZ  : min;
            max = max > tNearZ ? tNearZ : max;
        }else{
            min = min < tNearZ ? tNearZ : min;
            max = max > tFarZ  ? tFarZ  : max;
        }

        return min < max;
    }

private:
    const T minX;
    const T maxX;
    const T minY;
    const T maxY;
    const T minZ;
    const T maxZ;
    const Point3<T> center;
};