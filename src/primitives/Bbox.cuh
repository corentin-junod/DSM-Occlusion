#pragma once

#include "Point3.cuh"
#include "Ray.cuh"
#include "../array/Array.cuh"

#include <iostream>

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox():minX(0), maxX(0), minY(0), maxY(0), minZ(0), maxZ(0), center(Point3<T>(0, 0, 0)){}

    __host__ __device__ Bbox(T minX, T maxX, T minY, T maxY, T minZ, T maxZ) :
        minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ){
            center = Point3<T>( (maxX+minX)/2, (maxY+minY)/2, (maxZ+minZ)/2 );
        }

    template<typename U> __host__ friend std::ostream& operator<<(std::ostream& os, const Bbox<U>& bbox);

    __host__ __device__ void setEnglobing(Point3<T>** points, int size, float margin){
        minX = maxX = points[0]->x;
        minY = maxY = points[0]->y;
        minZ = maxZ = points[0]->z;
        for(int i=0; i<size; i++){
            const Point3<T>* point = points[i];
            if(point->x < minX) minX = point->x;
            if(point->x > maxX) maxX = point->x;
            if(point->y < minY) minY = point->y;
            if(point->y > maxY) maxY = point->y;
            if(point->z < minZ) minZ = point->z;
            if(point->z > maxZ) maxZ = point->z;
        }
        minX -= margin;
        minY -= margin;
        minZ -= margin;
        maxX += margin;
        maxY += margin;
        maxZ += margin;
        center.x = (maxX+minX)/2;
        center.y = (maxY+minY)/2;
        center.z = (maxZ+minZ)/2;
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
    T minX;
    T maxX;
    T minY;
    T maxY;
    T minZ;
    T maxZ;
    Point3<T> center;
};


template<typename T>
std::ostream& operator<<(std::ostream& os, const Bbox<T>& bbox){
    os << "Bbox( min(" << bbox.minX << "," << bbox.minY << "," << bbox.minZ << 
              ") max(" << bbox.maxX << "," << bbox.maxY << "," << bbox.maxZ << ") )";
    return os;
};