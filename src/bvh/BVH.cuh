#pragma once

#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../array/Array.cuh"
#include "../utils/utils.cuh"
#include <cstdio>
#include <ostream>

#define TILE_SIZE 0.5 

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
    Point3<T>** head;
    Point3<T>** tail;
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
    __host__ BVH(const bool useGPU, const unsigned int nbPixels): useGPU(useGPU) {
        bvhNodes       = (BVHNode<float>*)       allocMemory(2*nbPixels, sizeof(BVHNode<float>),       useGPU);
        bboxMemory     = (Bbox<float>*)          allocMemory(2*nbPixels, sizeof(Bbox<float>),          useGPU);
        elementsMemory = (Array<Point3<float>>*) allocMemory(2*nbPixels, sizeof(Array<Point3<float>>), useGPU);
        stackMemory    = (ArraySegment<float>*)  allocMemory(nbPixels,   sizeof(ArraySegment<float>),  useGPU);
        workingBuffer  = (Point3<float>**)       allocMemory(nbPixels,   sizeof(Point3<float>*),       useGPU);
        
    }

    __host__ void freeMemoryAfterBuild(){
        freeMemory(stackMemory,   useGPU);
        freeMemory(workingBuffer, useGPU);
    } 
    
    __host__ void free(){
        freeMemory(elementsMemory, useGPU);
        freeMemory(bboxMemory,     useGPU);
        freeMemory(bvhNodes,       useGPU);
    }

    __host__ __device__ void operator=(const BVH<T>& other){
        useGPU         = other.useGPU;
        bvhNodes       = other.bvhNodes;
        bboxMemory     = other.bboxMemory;
        elementsMemory = other.elementsMemory;
        stackMemory    = other.stackMemory;
        workingBuffer  = other.workingBuffer;
        nbNodes        = other.nbNodes;
        nbLeaves       = other.nbLeaves;
    }

    __host__ void printInfos(){std::cout<<"BVH : \n   Nodes : "<<nbNodes<<"\n   Leaves : "<<nbLeaves<<'\n';}
    __host__ __device__ int size() const {return nbNodes;}

    __host__ __device__ float getLighting(const Ray<T> ray, BVHNode<T>** buffer) const { 
        ray.getDirection().normalize();
        unsigned int bufferSize = 0;
        buffer[bufferSize++] = &bvhNodes[0];

        while(bufferSize > 0){
            BVHNode<T>* node = buffer[--bufferSize];

            if(node != nullptr && node->bbox->intersects(ray)){
                if(node->elements != nullptr){
                    for(const Point3<T>& point : *node->elements){
                        if(point != ray.getOrigin() && BVH::intersectBox(point, ray, TILE_SIZE/2)){
                            return 0;
                        }
                    }
                }
                buffer[bufferSize++] = node->left;
                buffer[bufferSize++] = node->right;
            }
        }
        return 1;
    }

    __host__ __device__ void build(Array<Point3<T>*>& points) {
        const float margin = TILE_SIZE/2;

        int BVHNodeCounter = 0;
        int bboxCounter = 0;

        Stack<T> stack = Stack<T>{stackMemory, 0};

        BVHNode<T>* root = new (&bvhNodes[BVHNodeCounter++]) BVHNode<T>();

        Point3<T>** begin = points.begin();
        Point3<T>** end   = points.end();
        stack.push(ArraySegment<T>{begin, end, root});
        
        while(stack.size > 0){
            nbNodes++;
            ArraySegment curSegment = stack.pop();
            const unsigned int curSize = curSegment.tail-curSegment.head;

            Bbox<T>* bbox = new (&bboxMemory[bboxCounter++]) Bbox<T>();
            bbox->setEnglobing(curSegment.head, curSize, margin);
            curSegment.node->bbox = bbox;

            if(curSize < bboxMaxSize){
                curSegment.node->elements = new (&elementsMemory[nbLeaves++]) Array<Point3<T>>(*curSegment.head, curSize);
            }else{

                const unsigned int splitIndex = BVH::split(curSegment.head, curSize, bbox);
                Point3<T>** middle = &(curSegment.head[splitIndex]);

                curSegment.node->left  = new (&bvhNodes[BVHNodeCounter++]) BVHNode<T>();
                curSegment.node->right = new (&bvhNodes[BVHNodeCounter++]) BVHNode<T>();

                stack.push(ArraySegment<T>{curSegment.head, middle, curSegment.node->left});
                stack.push(ArraySegment<T>{middle, curSegment.tail, curSegment.node->right});
            }
        }
    }

private:

    const bool useGPU;
    unsigned int nbNodes = 0;
    unsigned int nbLeaves = 0;
    const unsigned int bboxMaxSize = 5;

    BVHNode<float>*       bvhNodes;
    Bbox<float>*          bboxMemory;
    Array<Point3<float>>* elementsMemory;
    ArraySegment<float>*  stackMemory;
    Point3<float>**       workingBuffer;


    __host__ __device__ bool intersectBox(const Point3<T>& top, const Ray<T>& ray, const float margin) const {
        const Vec3<T>& rayDir = ray.getDirection();
        const Point3<T>& rayOrigin = ray.getOrigin();

        float min, max;

        const float xInverse = 1 / rayDir.x;
        const float tNearX = (top.x - margin - rayOrigin.x) * xInverse;
        const float tFarX  = (top.x + margin - rayOrigin.x) * xInverse;

        if(tNearX > tFarX){
            min = tFarX;
            max = tNearX;
        }else{
            min = tNearX;
            max = tFarX;
        }
        
        const float yInverse = 1 / rayDir.y;
        const float tNearY = (top.y - margin - rayOrigin.y) * yInverse;
        const float tFarY  = (top.y + margin - rayOrigin.y) * yInverse;

        if(tNearY > tFarY){
            min = min < tFarY  ? tFarY  : min;
            max = max > tNearY ? tNearY : max;
        }else{
            min = min < tNearY ? tNearY : min;
            max = max > tFarY  ? tFarY  : max;
        }

        if(max < min && min > 0) return false;

        const float zInverse = 1 / rayDir.z;
        const float tNearZ = (-rayOrigin.z) * zInverse;
        const float tFarZ  = (top.z - rayOrigin.z) * zInverse;

       if(tNearZ > tFarZ){
            min = min < tFarZ  ? tFarZ  : min;
            max = max > tNearZ ? tNearZ : max;
        }else{
            min = min < tNearZ ? tNearZ : min;
            max = max > tFarZ  ? tFarZ  : max;
        }

        return min < max && min > 0;
    }

    __host__ __device__ bool intersectSphere(const Point3<T>& top, const Ray<T>& ray, const float radius) const {
        const Point3<T> center = Point3<T>(top.x, top.y, top.z-TILE_SIZE/2);
        const T radius_squared = radius*radius;
        const Vec3<T> d_co = ray.getOrigin() - center;
        const T d_co_norm_sqr = d_co.getNormSquared();
        if(d_co_norm_sqr <= radius_squared) return false;
        const T tmp = ray.getDirection().dot(d_co);
        const T delta = tmp*tmp - (d_co_norm_sqr - radius_squared);
        const T t = -tmp-sqrt(delta);
        return delta >= 0 && t > 0;
    }

    __host__ __device__ int split(Point3<T>** points, unsigned int size, const Bbox<T>* bbox) const {
        const T dx = bbox->getEdgeLength('X');
        const T dy = bbox->getEdgeLength('Y');
        //const T dz = bbox->getEdgeLength('Z');
        const Point3<T> center = bbox->getCenter();

        int nbLeft  = 0;
        int nbRight = 0;


        for(int i=0; i<size; i++){
            Point3<T>* const point = points[i];
            if(dx>=dy /*&& dx>=dz*/){
                if(point->x < center.x){
                    workingBuffer[nbLeft++] = point;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }else /*if(dy>=dx && dy>=dz)*/{
                if(point->y < center.y){
                    workingBuffer[nbLeft++] = point;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }/*else{
                if(point->z < center.z){
                    workingBuffer[nbLeft++] = point;
                }else{
                    workingBuffer[size-nbRight-1] = point;
                    nbRight++;
                }
            }*/
        }
        for(int i=0; i<size; i++){
            points[i] = workingBuffer[i];
        }
        return nbLeft;
    }

};