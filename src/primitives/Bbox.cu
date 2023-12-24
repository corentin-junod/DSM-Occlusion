#include "Bbox.cuh"

#include "limits"
#include <algorithm>
#include <stdexcept>

constexpr int MAX_INT = std::numeric_limits<int>::max();
constexpr int MIN_INT = std::numeric_limits<int>::min();

template<typename T>
Bbox<T>::Bbox(int minX, int maxX, int minY, int maxY, int minZ, int maxZ) :
    minX(minX), maxX(maxX), minY(minY), maxY(maxY), minZ(minZ), maxZ(maxZ){}

template<typename T>
Bbox<T> Bbox<T>::getEnglobing(std::vector<Point3<T>*> objList){
    int minX = MAX_INT, minY = MAX_INT, minZ = MAX_INT;
    int maxX = MIN_INT, maxY = MIN_INT, maxZ = MIN_INT;
    for(const Point3<T>* const point : objList){
        if(point->x < minX) minX = point->x;
        if(point->x > maxX) maxX = point->x;
        if(point->y < minY) minY = point->y;
        if(point->y > maxY) maxY = point->y;
        if(point->z < minZ) minZ = point->z;
        if(point->z > maxZ) maxZ = point->z;
    }
    return Bbox(minX, maxX, minY, maxY, minZ, maxZ);
}

template<typename T>
Bbox<T> Bbox<T>::merge(const Bbox<T>* const other) const {
    const int newMinX = std::min(minX, other->minX);
    const int newMinY = std::min(minY, other->minY);
    const int newMinZ = std::min(minZ, other->minZ);
    const int newMaxX = std::max(maxX, other->maxX);
    const int newMaxY = std::max(maxY, other->maxY);
    const int newMaxZ = std::max(maxZ, other->maxZ);
    return Bbox(newMinX, newMaxX, newMinY, newMaxY, newMinZ, newMaxZ);
}

template<typename T>
int Bbox<T>::getEdgeLength(const int axis) const {
    switch (axis) {
        case 0 : return maxX - minX;
        case 1 : return maxY - minY;
        case 2 : return maxZ - minZ;
        default: throw std::invalid_argument("Invalid axis, value can be 0, 1 or 2");
    }
}

template<typename T>
bool Bbox<T>::intersects(const Ray<T>& ray){
    float min, max;

    const Vec3<T>* rayDir = ray.getDirection();
    const Point3<T>* rayOrigin = ray.getOrigin();

    const float xInverse = 1 / rayDir->x;
    const float tNearX = (minX - rayOrigin->x) * xInverse;
    const float tFarX  = (maxX - rayOrigin->x) * xInverse;

    if(tNearX > tFarX){
        min = tFarX;
        max = tNearX;
    }else{
        min = tNearX;
        max = tFarX;
    }

    const float yInverse = 1 / rayDir->y;
    const float tNearY = (minY - rayOrigin->y) * yInverse;
    const float tFarY  = (maxY - rayOrigin->y) * yInverse;

    if(tNearY > tFarY){
        min = min < tFarY  ? tFarY  : min;
        max = max > tNearY ? tNearY : max;
    }else{
        min = min < tNearY ? tNearY : min;
        max = max > tFarY  ? tFarY  : max;
    }

    if(max < min) return false;

    const float zInverse = 1 / rayDir->z;
    const float tNearZ = (minZ - rayOrigin->z) * zInverse;
    const float tFarZ  = (maxZ - rayOrigin->z) * zInverse;

    if(tNearZ > tFarZ){
        min = min < tFarZ  ? tFarZ  : min;
        max = max > tNearZ ? tNearZ : max;
    }else{
        min = min < tNearZ ? tNearZ : min;
        max = max > tFarZ  ? tFarZ  : max;
    }

    return min < max;
}