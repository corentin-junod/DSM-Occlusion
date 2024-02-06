#pragma once

#include "Point3.cuh"
#include "Ray.cuh"

#include <iostream>
#include <math.h>

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox():minX(0), maxX(0), minY(0), maxY(0), /*minZ(0),*/ maxZ(0){}

    __host__ __device__ Bbox(T minX, T maxX, T minY, T maxY, /*T minZ,*/ T maxZ) :
        minX(minX), maxX(maxX), minY(minY), maxY(maxY), /*minZ(minZ),*/ maxZ(maxZ){}

    template<typename U> __host__ friend std::ostream& operator<<(std::ostream& os, const Bbox<U>& bbox);

    __host__ __device__ void setEnglobing(Point3<T>** points, int size, T margin){
        minX = maxX = points[0]->x;
        minY = maxY = points[0]->y;
        /*minZ = */maxZ = points[0]->z;
        for(int i=0; i<size; i++){
            const Point3<T>* point = points[i];
            if(point->x < minX) minX = point->x;
            if(point->x > maxX) maxX = point->x;
            if(point->y < minY) minY = point->y;
            if(point->y > maxY) maxY = point->y;
            //if(point->z < minZ) minZ = point->z;
            if(point->z > maxZ) maxZ = point->z;
        }

        minX -= margin;
        minY -= margin;
        //minZ = 0;//-= margin*2;
        maxX += margin;
        maxY += margin;
        maxZ = maxZ;
    }

    __host__ __device__ T getEdgeLength(const char axis) const {
        switch (axis) {
            case 'X' : return maxX - minX;
            case 'Y' : return maxY - minY;
            default  : return maxZ /*- minZ*/;
        }
    }

    __host__ __device__ const Point3<T> getCenter() const {return Point3<T>( (maxX+minX)/TWO, (maxY+minY)/TWO, (maxZ/*+minZ*/)/TWO );}

    __host__ bool intersects(const Ray<Float>& ray) const {
        Float min, max;

        const Vec3<T>& rayDir = ray.getDirection();
        const Point3<T>& rayOrigin = ray.getOrigin();

        const Float xInverse = 1 / rayDir.x;
        const Float tNearX = (minX - rayOrigin.x) * xInverse;
        const Float tFarX  = (maxX - rayOrigin.x) * xInverse;

        if(tNearX > tFarX){
            min = tFarX;
            max = tNearX;
        }else{
            min = tNearX;
            max = tFarX;
        }
        
        const Float yInverse = 1 / rayDir.y;
        const Float tNearY = (minY - rayOrigin.y) * yInverse;
        const Float tFarY  = (maxY - rayOrigin.y) * yInverse;

        if(tNearY > tFarY){
            min = min < tFarY  ? tFarY  : min;
            max = max > tNearY ? tNearY : max;
        }else{
            min = min < tNearY ? tNearY : min;
            max = max > tFarY  ? tFarY  : max;
        }

        if(max < min) return false;

        const Float zInverse = 1 / rayDir.z;
        const Float tNearZ = (/*minZ */- rayOrigin.z) * zInverse;
        const Float tFarZ  = (maxZ - rayOrigin.z) * zInverse;

       if(tNearZ > tFarZ){
            min = min < tFarZ  ? tFarZ  : min;
            max = max > tNearZ ? tNearZ : max;
        }else{
            min = min < tNearZ ? tNearZ : min;
            max = max > tFarZ  ? tFarZ  : max;
        }

        return min < max && max > ZERO;
    }

    __device__ bool intersects(const Vec3<Float>& invRayDir, const Point3<Float>& rayOrigin) const {
        Float min, max;

        const Float tNearX = (minX - rayOrigin.x) * invRayDir.x;
        const Float tFarX  = (maxX - rayOrigin.x) * invRayDir.x;

        min = fminf(tNearX, tFarX);
        max = fmaxf(tNearX, tFarX);
        
        const Float tNearY = (minY - rayOrigin.y) * invRayDir.y;
        const Float tFarY  = (maxY - rayOrigin.y) * invRayDir.y;

        /*min = (tNearY > tFarY)*fmaxf(tFarY, min)+ (tNearY <= tFarY)*fmaxf(tNearY, min);
        max = (tNearY > tFarY)*fminf(tNearY, max)+ (tNearY <= tFarY)*fminf(tFarY, max);*/

        if(tNearY > tFarY){
            min = fmaxf(tFarY, min);
            max = fminf(tNearY, max);
        }else{
            min = fmaxf(tNearY, min);
            max = fminf(tFarY, max);
        }

        const Float tNearZ = (/*minZ*/ - rayOrigin.z) * invRayDir.z;
        const Float tFarZ  = (maxZ - rayOrigin.z) * invRayDir.z;

        if(tNearZ > tFarZ){
            min = fmaxf(tFarZ, min);
            max = fminf(tNearZ, max);
        }else{
            min = fmaxf(tNearZ, min);
            max = fminf(tFarZ, max);
        }

        return min < max && max > ZERO;
    }

private:
    T minX;
    T maxX;
    T minY;
    T maxY;
    //T minZ;
    T maxZ;
};


template<typename T>
std::ostream& operator<<(std::ostream& os, const Bbox<T>& bbox){
    os << "Bbox( min(" << bbox.minX << "," << bbox.minY << "," << /*bbox.minZ*/0 << 
              ") max(" << bbox.maxX << "," << bbox.maxY << "," << bbox.maxZ << ") )";
    return os;
};