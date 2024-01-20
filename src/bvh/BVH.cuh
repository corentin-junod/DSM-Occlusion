#pragma once

#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <cstdio>
#include <ostream>

#define TILE_SIZE 0.5 

constexpr unsigned int ELEMENTS_MAX_SIZE = 5; // Best is between 5 and 10

struct BVHNode{
    unsigned short int nbElements;
    unsigned int elementsIndex;
    Bbox<float> bbox;
    int leftIndex;
    int rightIndex;
    __host__ __device__ BVHNode() : bbox(Bbox<float>()), leftIndex(-1), rightIndex(-1), elementsIndex(0), nbElements(0) {}
};

template <typename T>
struct ArraySegment{
    Point3<T>** head;
    Point3<T>** tail;
    BVHNode* node;
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
    __host__ BVH(const unsigned int nbPixels): nbPixels(nbPixels) {
        bvhNodes       = (BVHNode*)       calloc(2*nbPixels, sizeof(BVHNode));
        elementsMemory = (Point3<float>*)        calloc(2*nbPixels, sizeof(Point3<float>)); // could be 1"nbPixels
        stackMemory    = (ArraySegment<float>*)  calloc(nbPixels,   sizeof(ArraySegment<float>));
        workingBuffer  = (Point3<float>**)       calloc(nbPixels,   sizeof(Point3<float>*));
    }

    __host__ void freeAfterBuild(){
        free(stackMemory);
        free(workingBuffer);
        stackMemory   = nullptr;
        workingBuffer = nullptr;
    }
    
    __host__ void freeAllMemory(){
        free(elementsMemory);
        free(bvhNodes);
    }

    __host__ BVH<T>* toGPU() const {
        BVHNode*  bvhNodesGPU       = (BVHNode*) allocGPU(2*nbPixels, sizeof(BVHNode));
        Point3<float>*   elementsMemoryGPU = (Point3<float>*)  allocGPU(2*nbPixels, sizeof(Point3<float>));
        
        checkError(cudaMemcpy(bvhNodesGPU,       bvhNodes,       2*nbPixels*sizeof(BVHNode), cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(elementsMemoryGPU, elementsMemory, 2*nbPixels*sizeof(Point3<float>),  cudaMemcpyHostToDevice));

        ArraySegment<float>* stackMemoryGPU = nullptr;
        if(stackMemory != nullptr){
            stackMemoryGPU    = (ArraySegment<float>*)  allocGPU(nbPixels,   sizeof(ArraySegment<float>));
            checkError(cudaMemcpy(stackMemoryGPU, stackMemory, nbPixels*sizeof(ArraySegment<float>), cudaMemcpyHostToDevice));
        }

        Point3<float>** workingBufferGPU = nullptr;
        if(workingBuffer != nullptr){
            workingBufferGPU = (Point3<float>**) allocGPU(nbPixels, sizeof(Point3<float>*));
            checkError(cudaMemcpy(workingBufferGPU, workingBuffer, nbPixels*sizeof(Point3<float>*), cudaMemcpyHostToDevice));
        }

        BVH<T> tmp = BVH<T>(nbPixels);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        tmp.bvhNodes       = bvhNodesGPU;
        tmp.elementsMemory = elementsMemoryGPU;
        tmp.stackMemory    = stackMemoryGPU;
        tmp.workingBuffer  = workingBufferGPU;
        BVH<T>* replica = (BVH<T>*) allocGPU(sizeof(BVH<T>));
        checkError(cudaMemcpy(replica, &tmp, sizeof(BVH<T>), cudaMemcpyHostToDevice));
        return replica;
    }

    __host__ void fromGPU(BVH<T>* replica){
        BVH<T> tmp = BVH<T>(nbPixels);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        checkError(cudaMemcpy(&tmp, replica, sizeof(BVH<T>), cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(bvhNodes,       tmp.bvhNodes,       2*nbPixels*sizeof(BVHNode),       cudaMemcpyDeviceToHost));
        checkError(cudaMemcpy(elementsMemory, tmp.elementsMemory, 2*nbPixels*sizeof(Point3<float>), cudaMemcpyDeviceToHost));
        
        freeGPU(tmp.bvhNodes);
        freeGPU(tmp.elementsMemory);

        if(stackMemory != nullptr){
            checkError(cudaMemcpy(stackMemory, tmp.stackMemory, nbPixels*sizeof(ArraySegment<float>), cudaMemcpyDeviceToHost));
            freeGPU(tmp.stackMemory);
        }
        if(workingBuffer != nullptr){
            checkError(cudaMemcpy(workingBuffer, tmp.workingBuffer, nbPixels*sizeof(Point3<float>*), cudaMemcpyDeviceToHost));
            freeGPU(tmp.workingBuffer);
        }

        nbNodes = tmp.nbNodes;

        freeGPU(replica);
    }

    __host__ void printInfos(){std::cout<<"BVH : \n   Nodes : "<<nbNodes<<"\n";}
    __host__ __device__ int size() const {return nbNodes;}


    __host__ __device__ float getLighting(const Point3<T>& origin, const Vec3<T>& dir, int* const buffer) const {
        unsigned short bufferSize = 0;
        buffer[bufferSize++] = 0;

        while(bufferSize > 0){
            const BVHNode node = bvhNodes[buffer[--bufferSize]];
            if(node.bbox.intersects(dir, origin)){
                for(unsigned short i=0; i<node.nbElements; i++){
                    const Point3<T> point = elementsMemory[node.elementsIndex+i];
                    if(point != origin && intersectBox(point, dir, origin, TILE_SIZE/2)){
                        return 0;
                    }
                }

                if(node.leftIndex >= 0){
                    buffer[bufferSize++] = node.leftIndex;
                }

                if(node.rightIndex >= 0){
                    buffer[bufferSize++] = node.rightIndex;
                }
            }
        }
        return 1;
    }

    __host__ __device__ void build(Array2D<Point3<T>*>& points) {
        const float margin = TILE_SIZE/2;

        unsigned int BVHNodeCounter  = 0;
        unsigned int elementsCounter = 0;

        Stack<T> stack = Stack<T>{stackMemory, 0};

        BVHNode* root = new (&bvhNodes[BVHNodeCounter++]) BVHNode();
        stack.push(ArraySegment<T>{points.begin(), points.end(), root});
        
        while(stack.size > 0){
            ArraySegment curSegment = stack.pop();
            const unsigned int curSize = curSegment.tail-curSegment.head;

            curSegment.node->bbox.setEnglobing(curSegment.head, curSize, margin);
            
            if(curSize < ELEMENTS_MAX_SIZE){
                for(int i=0; i<curSize; i++){
                    elementsMemory[elementsCounter+i] = *(curSegment.head[i]);
                }
                curSegment.node->elementsIndex = elementsCounter;
                curSegment.node->nbElements = curSize;
                elementsCounter += curSize;
            }else{

                const unsigned int splitIndex = split(curSegment.head, curSize,  curSegment.node->bbox);
                Point3<T>** middle = &(curSegment.head[splitIndex]);

                curSegment.node->leftIndex  = BVHNodeCounter;
                BVHNode* leftNode = new (&bvhNodes[BVHNodeCounter++]) BVHNode();
                curSegment.node->rightIndex = BVHNodeCounter;
                BVHNode* rightNode = new (&bvhNodes[BVHNodeCounter++]) BVHNode();
                
                stack.push(ArraySegment<T>{curSegment.head, middle, leftNode});
                stack.push(ArraySegment<T>{middle, curSegment.tail, rightNode});
            }

            nbNodes++;
        }
    }

private:

    const unsigned int nbPixels;
    unsigned int nbNodes = 0;

    BVHNode*             bvhNodes;
    Point3<float>*       elementsMemory;
    ArraySegment<float>* stackMemory;
    Point3<float>**      workingBuffer;


    __host__ __device__ bool intersectBox(const Point3<T>& top, const Vec3<T>& rayDir, const Point3<T>& rayOrigin, const float margin) const {
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
        ray.getDirection().normalize();
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

    __host__ __device__ int split(Point3<T>** points, unsigned int size, const Bbox<T>& bbox) const {
        const T dx = bbox.getEdgeLength('X');
        const T dy = bbox.getEdgeLength('Y');
        //const T dz = bbox->getEdgeLength('Z');
        const Point3<T> center = bbox.getCenter();

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