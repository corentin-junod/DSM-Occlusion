#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../array/Array.cuh"

#include <iostream>

template<typename T>
struct BVHNode{
    Bbox<T>* bbox;
    BVHNode* left;
    BVHNode* right;
    Array<Point3<T>>* elements;
    __host__ __device__ BVHNode() : bbox(nullptr), left(nullptr), right(nullptr), elements(nullptr) {}
};

template <typename T>
struct ArraySegment{
    Point3<T>* head;
    Point3<T>* tail;
    BVHNode<T>* node;
};

template <typename T>
struct Stack {
    ArraySegment<T>* data;
    unsigned int size;
    __host__ __device__ void push(ArraySegment<T> value) {data[size++] = value;}
    __host__ __device__ ArraySegment<T> pop() {return data[--size];}
};



template<typename T>
class BVH{

public:
    __host__ __device__ BVH(Array<Point3<T>>& points, ArraySegment<T>* stackMemory, Point3<T>** workingBufferPMemory) : root(BVH::build(points, stackMemory, workingBufferPMemory)){}
    __host__ __device__ int size() const {return nbElements;}
    __host__ __device__ BVHNode<T>* getRoot() const {return root;}

    //__host__ __device__ bool isIntersectingIter(   Ray<T>& ray, std::vector<BVHNode<T>*>& buffer) const { return BVH<T>::isIntersectingIterate(ray, root, buffer);}
             __device__ bool isIntersectingIterGPU(Ray<T>& ray, BVHNode<float>** buffer) const { return BVH<T>::isIntersectingIterateGPU(ray, root, buffer);}
    __host__ __device__ bool isIntersectingRec(    Ray<T>& ray) const { return BVH<T>::isIntersectingRecursive(ray, root);}

private:

    BVHNode<T>* const root;
    int nbElements = 0;

     __host__ __device__ BVHNode<T>* build(Array<Point3<T>>& points, ArraySegment<T>* stackMemory, Point3<T>** workingBufferPMemory){
        Array<Point3<T>*> workingBuffer = Array<Point3<T>*>(workingBufferPMemory, points.size());
        BVHNode<T>* newRoot = BVH::buildRecursive(points, workingBuffer, points.size(), stackMemory);
        return newRoot;
     }

    __host__ __device__ BVHNode<T>* buildRecursive(Array<Point3<T>>& points, Array<Point3<T>*> workingBuffer, unsigned int size, ArraySegment<T>* stackMemory) {

        Stack<T> stack = Stack<T>{stackMemory, 0};

        BVHNode<T>* root = new BVHNode<T>();

        Point3<T>* begin = points.begin();
        stack.push(ArraySegment<T>{begin, begin+size, root});

        while(stack.size > 0){
            nbElements++;
            ArraySegment curSegment = stack.pop();
            int curSize = curSegment.tail-curSegment.head;


            Bbox<T>* bbox = new Bbox<T>();
            bbox->setEnglobing(curSegment.head, curSize);
            curSegment.node->bbox = bbox;

            if(curSize < 10){
                curSegment.node->left  = nullptr;
                curSegment.node->right = nullptr;
                curSegment.node->elements = new Array<Point3<T>>(curSegment.head, curSize);
            }else{
                int splitIndex = BVH::split(curSegment.head, workingBuffer, curSize, *bbox);

                curSegment.node->left  = new BVHNode<T>();
                curSegment.node->right = new BVHNode<T>();
                curSegment.node->elements = nullptr;

                stack.push(ArraySegment<T>{curSegment.head, &curSegment.head[splitIndex], curSegment.node->left});
                stack.push(ArraySegment<T>{&curSegment.head[splitIndex], curSegment.tail, curSegment.node->right});
            }
        }
        
        return root;
    }

    __host__ __device__ int split(Point3<T>* points, Array<Point3<T>*> workingBuffer, unsigned int size, const Bbox<T>& bbox) const {
        const T dx = bbox.getEdgeLength('X');
        const T dy = bbox.getEdgeLength('Y');
        const T dz = bbox.getEdgeLength('Z');
        const Point3<T> center = bbox.getCenter();

        int nbLeft  = 0;
        int nbRight = 0;

        for(int i=0; i<size; i++){
            Point3<T>* const point = &points[i];
            if(dx>=dy && dx>=dz){
                if(point->x < center.x){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }else if(dy>=dx && dy>=dz){
                if(point->y < center.y){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }else{
                if(point->z < center.z){
                    workingBuffer[nbLeft] = point;
                    nbLeft++;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }
        }
        for(int i=0; i<size; i++){
            points[i] = *workingBuffer[i];
        }
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

            if(currentNode != nullptr && currentNode->bbox->intersects(ray)){
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