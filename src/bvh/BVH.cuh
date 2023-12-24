#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"

#include <vector>

template<typename T>
struct BVHNode{
    Bbox<T> bbox;
    const BVHNode* left;
    const BVHNode* right;
    std::vector<Point3<T>*>* elements;
};

template<typename T>
class BVH{

public:
    __host__ __device__ BVH(std::vector<Point3<T>*>& points);
    __host__ __device__ bool isIntersecting(Ray<T>& ray) const;

private:

    const BVHNode<T>* const root;
    const BVHNode<T>* buildRecursive(std::vector<Point3<T>*>& points) const;
    bool isIntersectingRecursive(Ray<T>& ray, BVHNode<T>* node) const;
    void split(std::vector<Point3<T>*>& points, Bbox<T>& parent, std::vector<Point3<T>*>* out_left, std::vector<Point3<T>*>* out_right) const;

};