#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../array/Array.cuh"

#include <iostream>

template<typename T>
struct BVHNode{
    Bbox<T> bbox;
    BVHNode* left;
    BVHNode* right;
    Array<Point3<T>>* elements;
};

template<typename T>
struct ArraySection{
    Point3<T>** head;
    int size;
};

template<typename T>
class BVH{

public:
    __host__ __device__ BVH(Array<Point3<T>>& points) : root(BVH::build(points)){}
    __host__ __device__ int size() const {return nbElements;}
    __host__ __device__ BVHNode<T>* getRoot() const {return root;}

    //__host__ __device__ bool isIntersectingIter(   Ray<T>& ray, std::vector<BVHNode<T>*>& buffer) const { return BVH<T>::isIntersectingIterate(ray, root, buffer);}
             __device__ bool isIntersectingIterGPU(Ray<T>& ray, BVHNode<float>** buffer) const { return BVH<T>::isIntersectingIterateGPU(ray, root, buffer);}
    __host__ __device__ bool isIntersectingRec(    Ray<T>& ray) const { return BVH<T>::isIntersectingRecursive(ray, root);}

private:

    BVHNode<T>* const root;
    int nbElements = 0;

     __host__ __device__ BVHNode<T>* build(Array<Point3<T>>& points){
        Point3<T>* workingBuffer; 
        new cudaMalloc(&workingBuffer, points.size()*sizeof(Point3<T>*));
        BVHNode<T>* newRoot = BVH::buildRecursive(points.begin(), workingBuffer, 0, points.size());
        cudaFree(workingBuffer);
        return newRoot;
     }

    __host__ __device__ BVHNode<T>* buildRecursive(Point3<T>** points, Point3<T>** workingBuffer, unsigned int size) {
/*
        ArraySection* sectionsToSort;
        new cudaMalloc(&sectionsToSort, size*sizeof(ArraySection));

        sequencesToSort[0] = {points[0], size};
        sequencesSize = 1;
        sequencesIndex = 0;

        while(sequencesSize > 0){
            ArraySection currentSection = sectionsToSort[sequencesIndex];
            Bbox<T> bbox = Bbox<T>::getEnglobing(currentSection.head, currentSection.size);
            int splitIndex = BVH::split(currentSection.head, workingBuffer, currentSection.size, bbox);

            sequencesToSort[sequencesIndex] = ...
            sequencesIndex++;
            sequencesToSort[sequencesIndex] = ...
            sequencesIndex++;



            BVHNode<T>* head = 
        }

        


        nbElements++;

    

        if(points.size() < 5){
            return new BVHNode<T>{bbox, nullptr, nullptr, &points};
        }

        int nbLeft = 0;
        int nbRight = 0;
        BVH::split(points, bbox, workingBuffer, nbLeft, nbRight);


        Array<Point3<T>> left  = Array<Point3<T>>(&leftVector[0],  leftVector.size());
        Array<Point3<T>> right = Array<Point3<T>>(&rightVector[0], rightVector.size());

        return new BVHNode<T>{bbox, BVH::buildRecursive(left, workingBuffer), BVH::buildRecursive(right, workingBuffer), nullptr};*/
    }

    __host__ __device__ int split(Point3<T>** points, Point3<T>** workingBuffer, unsigned int size, const Bbox<T>& bbox) const {
        const T dx = bbox.getEdgeLength('X');
        const T dy = bbox.getEdgeLength('Y');
        const T dz = bbox.getEdgeLength('Z');
        const Point3<T> center = bbox.getCenter();

        int nbLeft  = 0;
        int nbRight = 0;

        for(int i=0; i<size; i++){
            const Point3<T>* const point = points[i];
            if(dx>=dy && dx>=dz){
                if(point.x < center.x){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight] = point;
                    nbRight++;
                }
            }else if(dy>=dx && dy>=dz){
                if(point.y < center.y){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight] = point;
                    nbRight++;
                }
            }else{
                if(point.z < center.z){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight] = point;
                    nbRight++;
                }
            }
        }
        Point3<T>* tmp = *workingBuffer;
        *workingBuffer = *points;
        *points = tmp;
        return nbLeft;
    }

    __host__ __device__ bool isIntersectingRecursive(const Ray<T>& ray, const BVHNode<T>* const node, int depth=0) const {
        if(depth < 0 && node != nullptr && node->bbox.intersects(ray)){
            if(node->elements != nullptr){
                return true; // TODO Handle this case
            }
            return BVH<T>::isIntersectingRecursive(ray, node->left, depth+1) || BVH<T>::isIntersectingRecursive(ray, node->right, depth+1);
        }
        return false;
    }

    /*__host__ bool isIntersectingIterate(const Ray<T>& ray, BVHNode<T>* const node, std::vector<BVHNode<T>*>& buffer) const {
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
    }*/

    __device__ bool isIntersectingIterateGPU(const Ray<T>& ray, BVHNode<T>* const node, BVHNode<float>** const buffer) const {
        buffer[0] = node;
        unsigned int nbElemInBuffer = 1;

        while(nbElemInBuffer > 0){

            BVHNode<T>* currentNode = buffer[nbElemInBuffer-1];
            nbElemInBuffer--;

            if(currentNode != nullptr && currentNode->bbox.intersects(ray)){
                return true;
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