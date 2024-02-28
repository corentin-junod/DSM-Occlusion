#pragma once

#include "Point3.cuh"
#include <iostream>

template<typename T>
class Bbox{
public:

    __host__ __device__ Bbox():minX(0), maxX(0), minY(0), maxY(0), /*minZ(0),*/ maxZ(0){}

    __host__ __device__ Bbox(T minX, T maxX, T minY, T maxY, /*T minZ,*/ T maxZ) :
        minX(minX), maxX(maxX), minY(minY), maxY(maxY), /*minZ(minZ),*/ maxZ(maxZ){}

    template<typename U> __host__ friend std::ostream& operator<<(std::ostream& os, const Bbox<U>& bbox);

    __host__ void setEnglobing(Point3<T>** const points, const uint size, const T margin){
        minX = maxX = points[0]->x;
        minY = maxY = points[0]->y;
        /*minZ = */maxZ = points[0]->z;
        for(uint i=1; i<size; i++){
            const Point3<T>* const point = points[i];
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
        //maxZ = maxZ;
    }

    __host__ T getEdgeLength(const char axis) const {
        switch (axis) {
            case 'X' : return maxX - minX;
            case 'Y' : return maxY - minY;
            default  : return maxZ /*- minZ*/;
        }
    }

    __host__ __device__ const Point3<T> getCenter() const {
        return Point3<T>( (maxX+minX)/2, (maxY+minY)/2, (maxZ/*+minZ*/)/2 );
    }

    __device__ bool intersects(const Vec3<float>& invRayDir, const Point3<float>& rayOrigin) const {
        float min, max;

        const float tX1 = (minX - rayOrigin.x) * invRayDir.x;
        const float tX2  = (maxX - rayOrigin.x) * invRayDir.x;

        min = fminf(tX1, tX2);
        max = fmaxf(tX1, tX2);
        
        const float tY1 = (minY - rayOrigin.y) * invRayDir.y;
        const float tY2  = (maxY - rayOrigin.y) * invRayDir.y;

        const float tNearY = fminf(tY1, tY2);
        const float tFarY  = fmaxf(tY1, tY2);

        min = fmaxf(tNearY, min);
        max = fminf(tFarY, max);

        const float tZ1 = (/*minZ*/ - rayOrigin.z) * invRayDir.z;
        const float tZ2  = (maxZ - rayOrigin.z) * invRayDir.z;

        const float tNearZ = fminf(tZ1, tZ2);
        const float tFarZ  = fmaxf(tZ1, tZ2);

        min = fmaxf(tNearZ, min);
        max = fminf(tFarZ, max);

        return min < max && max > 0;
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