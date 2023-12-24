#include "BVH.cuh"

#include <algorithm>
#include <tuple>
#include <vector>

template<typename T>
BVH<T>::BVH(std::vector<Point3<T>*>& points) : root(BVH::buildRecursive(points)){}

template<typename T>
const BVHNode<T>* BVH<T>::buildRecursive(std::vector<Point3<T>*>& points) const{
    Bbox<T> parentBbox = Bbox<T>::getEnglobing(points);

    if(points.size() < 5){
        return new BVHNode<T>{parentBbox, nullptr, nullptr, &points};
    }

    std::vector<Point3<T>*> left;
    std::vector<Point3<T>*> right;
    BVH::split(points, parentBbox, &left, &right);

    return new BVHNode<T>{parentBbox, BVH::buildRecursive(left), BVH::buildRecursive(right), nullptr};
}

template<typename T>
void BVH<T>::split(std::vector<Point3<T>*>& points, Bbox<T>& parent, std::vector<Point3<T>*>* out_left, std::vector<Point3<T>*>* out_right) const{
    const int dx = parent.getEdgeLength(0);
    const int dy = parent.getEdgeLength(1);
    const int dz = parent.getEdgeLength(2);
    const int max = std::max({dx, dy, dz});
    const double splitValue = max/2.0;

    if(dx == max){
        for(Point3<T>* const point : points){
            if(point->x < splitValue) out_left->push_back(point);
            else out_right->push_back(point);
        }
    }else if(dy == max){
        for(Point3<T>* const point : points){
            if(point->y < splitValue) out_left->push_back(point);
            else out_right->push_back(point);
    }
    }else{
        for(Point3<T>* const point : points){
            if(point->z < splitValue) out_left->push_back(point);
            else out_right->push_back(point);
        }
    }
}

template<typename T>
bool BVH<T>::isIntersecting(Ray<T>& ray) const{
    return BVH<T>::isIntersectingRecursive(ray, root);
}

template<typename T>
bool BVH<T>::isIntersectingRecursive(Ray<T>& ray, BVHNode<T>* node) const{
    if(node->bbox.intersects(ray)){
        if(node->elements != nullptr){
            return true; // TODO Handle this case
        }
        return BVH<T>::isIntersectingRecursive(ray, node->left) || BVH<T>::isIntersectingRecursive(ray, node->right);
    }
    return false;
}