#pragma once

#include "Point3.cuh"
#include "Ray.cuh"
#include "../sizedArray/SizedArray.cuh"

#include <limits>
#include <algorithm>
#include <stdexcept>

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox(T minX, T maxX, T minY, T maxY, T minZ, T maxZ) :
        minX(minX), maxX(maxX), minY(minY), 
        maxY(maxY), minZ(minZ), maxZ(maxZ), 
        center(Point3<T>((maxX+minX)/2, (maxY+minY)/2, (maxZ+minZ)/2)){};

    __host__  static Bbox getEnglobing(SizedArray<Point3<T>>& objList){
        const int MAX_INT = std::numeric_limits<int>::max();
        const int MIN_INT = std::numeric_limits<int>::min();
        int minX = MAX_INT, minY = MAX_INT, minZ = MAX_INT;
        int maxX = MIN_INT, maxY = MIN_INT, maxZ = MIN_INT;
        for(Point3<T> point : objList){
            if(point.x < minX) minX = point.x;
            if(point.x > maxX) maxX = point.x;
            if(point.y < minY) minY = point.y;
            if(point.y > maxY) maxY = point.y;
            if(point.z < minZ) minZ = point.z;
            if(point.z > maxZ) maxZ = point.z;
        }
        return Bbox(minX, maxX, minY, maxY, minZ, maxZ);
    }

    __host__ __device__ Bbox merge(const Bbox* const other) const {
        const int newMinX = std::min(minX, other->minX);
        const int newMinY = std::min(minY, other->minY);
        const int newMinZ = std::min(minZ, other->minZ);
        const int newMaxX = std::max(maxX, other->maxX);
        const int newMaxY = std::max(maxY, other->maxY);
        const int newMaxZ = std::max(maxZ, other->maxZ);
        return Bbox(newMinX, newMaxX, newMinY, newMaxY, newMinZ, newMaxZ);
    }

    __host__ __device__ int getEdgeLength(const int axis) const {
        switch (axis) {
            case 0  : return maxX - minX;
            case 1  : return maxY - minY;
            default : return maxZ - minZ;
        }
    }

    __host__ __device__ Point3<T> getCenter() const {return center;}

    __host__ __device__ bool intersects(const Ray<T>& ray) const {
        float min, max;

        const Vec3<T> rayDir = ray.getDirection();
        const Point3<T> rayOrigin = ray.getOrigin();

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
    const int minX;
    const int maxX;
    const int minY;
    const int maxY;
    const int minZ;
    const int maxZ;
    const Point3<T> center;
};