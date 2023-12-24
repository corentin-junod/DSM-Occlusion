#include "Ray.cuh"

template<typename T>
Ray<T>::Ray(Point3<T>* origin, Vec3<T>* direction) : origin(origin), direction(direction){}

