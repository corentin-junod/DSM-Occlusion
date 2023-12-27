#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../array/Array.cuh"

#include <iostream>
#include <vector>

template<typename T>
struct BVHNode{
    Bbox<T> bbox;
    BVHNode* left;
    BVHNode* right;
    Array<Point3<T>>* elements;
};

template<typename T>
class BVH{

public:
    __host__ BVH(Array<Point3<T>>& points) : root(BVH::buildRecursive(points, 0)){}
    __host__ __device__ int size() const {return nbElements;}
    __host__ __device__ BVHNode<T>* getRoot() const {return root;}


    __host__ __device__ bool isIntersectingIter(   Ray<T>& ray, BVHNode<T>* currentRoot, std::vector<BVHNode<T>*>& buffer) const { return BVH<T>::isIntersectingIterate(ray, currentRoot, buffer);}
             __device__ bool isIntersectingIterGPU(Ray<T>& ray, BVHNode<T>* currentRoot, BVHNode<float>** buffer) const { return BVH<T>::isIntersectingIterateGPU(ray, currentRoot, buffer);}
    __host__ __device__ bool isIntersectingRec(    Ray<T>& ray, BVHNode<T>* currentRoot) const { return BVH<T>::isIntersectingRecursive(ray, currentRoot);}

    __host__ void copyNodes(BVHNode<T>* dest){copyNodesRecursive(dest, root, 0);};

private:

    BVHNode<T>* const root;
    int nbElements = 0;

    __host__ int copyNodesRecursive(BVHNode<T>* dest, BVHNode<T>* currentNode, int counter){
        int counterLeft  = 0;
        int counterRight = 0;
        int counterFinal = 0;

        if(currentNode->left != nullptr){
            counterLeft = counter + 1;
            counterRight = 1 + copyNodesRecursive(dest, currentNode->left, counterLeft);
            counterFinal = counterRight;
        }

        if(currentNode->right != nullptr){
            if(counterRight == 0) counterRight = counter + 1;
            counterFinal = 1 + copyNodesRecursive(dest, currentNode->right, counterRight);
        }

        dest[counter] = {
            currentNode->bbox, 
            &dest[counterLeft],
            &dest[counterRight],
            currentNode->elements
        };

        return counterFinal;
    }

    __host__ BVHNode<T>* buildRecursive(Array<Point3<T>>& points, int depth) {
        if(points.size() == 0){
            return nullptr;
        }
        nbElements++;

        Bbox<T> bbox = Bbox<T>::getEnglobing(points);

        if(points.size() < 3){
            return new BVHNode<T>{bbox, nullptr, nullptr, &points};
        }

        std::vector<Point3<T>> leftVector  = std::vector<Point3<T>>();
        std::vector<Point3<T>> rightVector = std::vector<Point3<T>>();
        BVH::split(points, bbox, leftVector, rightVector);

        Array<Point3<T>> left  = Array<Point3<T>>(&leftVector[0],  leftVector.size());
        Array<Point3<T>> right = Array<Point3<T>>(&rightVector[0], rightVector.size());

        return new BVHNode<T>{bbox, BVH::buildRecursive(left, depth+1), BVH::buildRecursive(right, depth+1), nullptr};
    }

    __host__ void split(const Array<Point3<T>>& points, const Bbox<T>& bbox, std::vector<Point3<T>>& out_left, std::vector<Point3<T>>& out_right) const {
        const T dx = bbox.getEdgeLength('X');
        const T dy = bbox.getEdgeLength('Y');
        const T dz = bbox.getEdgeLength('Z');
        const Point3<T> center = bbox.getCenter();

        if(dx>=dy && dx>=dz){
            for(Point3<T>& point : points){
                if(point.x < center.x) out_left.push_back(point);
                else out_right.push_back(point);
            }
        }else if(dy>=dx && dy>=dz){
            for(Point3<T>& point : points){
                if(point.y < center.y) out_left.push_back(point);
                else out_right.push_back(point);
        }
        }else{
            for(Point3<T>& point : points){
                if(point.z < center.z) out_left.push_back(point);
                else out_right.push_back(point);
            }
        }
    }

    __host__ __device__ bool isIntersectingRecursive(const Ray<T>& ray, const BVHNode<T>* const node) const {
        if(node != nullptr /*&& node->bbox.intersects(ray)*/){
            //if(node->elements != nullptr){
                return true; // TODO Handle this case
            //}
            //return BVH<T>::isIntersectingRecursive(ray, node->left) || BVH<T>::isIntersectingRecursive(ray, node->right);
        }
        return false;
    }

    __host__ bool isIntersectingIterate(const Ray<T>& ray, BVHNode<T>* const node, std::vector<BVHNode<T>*>& buffer) const {
        buffer.clear();
        buffer.push_back(node);

        while(buffer.size() > 0){

            BVHNode<T>* currentNode = buffer.back();
            buffer.pop_back();

            if(currentNode != nullptr && currentNode->bbox.intersects(ray)){
                if(currentNode->elements != nullptr){
                    return true; // TODO Handle this case
                }
                buffer.push_back(currentNode->left);
                buffer.push_back(currentNode->right);
            }
        }

        return false;
    }

    __device__ bool isIntersectingIterateGPU(const Ray<T>& ray, BVHNode<T>* const node, BVHNode<float>** buffer) const {
        buffer[0] = node;
        int nbElemInBuffer = 1;

        while(nbElemInBuffer > 0){

            BVHNode<T>* currentNode = buffer[nbElemInBuffer-1];
            nbElemInBuffer--;

            if(currentNode != nullptr && currentNode->bbox.intersects(ray)){
                if(currentNode->elements != nullptr){
                    return true; // TODO Handle this case
                }
                buffer[nbElemInBuffer] = currentNode->left;
                nbElemInBuffer++;
                buffer[nbElemInBuffer] = currentNode->right;
                nbElemInBuffer++;
            }
        }

        return false;
    }
};