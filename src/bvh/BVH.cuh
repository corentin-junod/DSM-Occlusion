#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../sizedArray/SizedArray.cuh"

#include <iostream>
#include <vector>

template<typename T>
struct BVHNode{
    Bbox<T> bbox;
    const BVHNode* left;
    const BVHNode* right;
    SizedArray<Point3<T>>* elements;
};

template<typename T>
class BVH{

public:
    __host__ __device__ BVH(SizedArray<Point3<T>>& points) : root(BVH::buildRecursive(points, 0)){};
    __host__ __device__ bool isIntersecting(Ray<T>& ray) const{ return BVH<T>::isIntersectingRecursive(ray, root, 0);};

private:

    const BVHNode<T>* const root;
    int depth = 0;

    __host__  const BVHNode<T>* buildRecursive(SizedArray<Point3<T>>& points, int depth) const {
        //std::cout << "Depth "<< depth <<" New iteration with " << points.getSize() << " points \n";

        if(points.getSize() == 0){
            return nullptr;
        }

        Bbox<T> parentBbox = Bbox<T>::getEnglobing(points);


        if(points.getSize() < 5){
            return new BVHNode<T>{parentBbox, nullptr, nullptr, &points};
        }

        std::vector<Point3<T>> leftVector;
        std::vector<Point3<T>> rightVector;
        BVH::split(points, parentBbox, &leftVector, &rightVector);

        SizedArray<Point3<T>> left  = SizedArray<Point3<T>>(&leftVector[0], leftVector.size());
        SizedArray<Point3<T>> right = SizedArray<Point3<T>>(&rightVector[0], rightVector.size());

        return new BVHNode<T>{parentBbox, BVH::buildRecursive(left, depth+1), BVH::buildRecursive(right,  depth+1), nullptr};
    }

    __host__  void split(SizedArray<Point3<T>>& points, Bbox<T>& parent, std::vector<Point3<T>>* out_left, std::vector<Point3<T>>* out_right) const {
        const int dx = parent.getEdgeLength(0);
        const int dy = parent.getEdgeLength(1);
        const int dz = parent.getEdgeLength(2);
        const Point3<T> center = parent.getCenter();

        if(dx>=dy && dx>=dz){
            for(const Point3<T> point : points){
                if(point.x < center.x) out_left->push_back(point);
                else out_right->push_back(point);
            }
        }else if(dy>=dx && dy>=dz){
            for(const Point3<T> point : points){
                if(point.y < center.y) out_left->push_back(point);
                else out_right->push_back(point);
        }
        }else{
            for(const Point3<T> point : points){
                if(point.z < center.z) out_left->push_back(point);
                else out_right->push_back(point);
            }
        }
    }

    __host__ bool isIntersectingRecursive(Ray<T>& ray, const BVHNode<T>* node, int depth) const {
        //std::cout << "Intersection depth " << depth << '\n';
        if(node != nullptr && node->bbox.intersects(ray)){
            if(node->elements != nullptr){
                return true; // TODO Handle this case
            }
            return BVH<T>::isIntersectingRecursive(ray, node->left, depth+1) || BVH<T>::isIntersectingRecursive(ray, node->right, depth+1);
        }
        return false;
    }
};