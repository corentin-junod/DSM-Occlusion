#pragma once

#include "../primitives/Bbox.cuh"
#include "../primitives/Ray.cuh"
#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <cstdio>
#include <iostream>
#include <ostream>
#include <vector>

constexpr byte ELEMENTS_MAX_SIZE = 4;

struct __align__(16) BVHNode{
    Bbox<float> bboxLeft  = Bbox<float>();
    Bbox<float> bboxRight = Bbox<float>();
    uint elementsIndex = 0;
    uint sizeLeft      = 0;
    uint sizeRight     = 0;
    byte nbElements    = 0;
    /*byte cacheIdLeft  = 255;
    byte cacheIdRight = 255;*/
};

struct ArraySegment{
    ArraySegment* parent;
    Point3<float>** head;
    Point3<float>** tail;
    BVHNode* node = nullptr;
};

class BVH{
public:
    __host__ BVH(const uint nbPixels, const float pixelSize): 
    nbPixels(nbPixels), pixelSize(pixelSize) {
        bvhNodes       = (BVHNode*)        calloc(2*nbPixels, sizeof(BVHNode));
        elementsMemory = (Point3<float>*)  calloc(nbPixels,   sizeof(Point3<float>));
        stackMemory    = (ArraySegment*)   calloc(nbPixels,   sizeof(ArraySegment));
        workingBuffer  = (Point3<float>**) calloc(nbPixels,   sizeof(Point3<float>*));
    }

    __host__ void freeAfterBuild() {
        free(stackMemory);
        free(workingBuffer);
        stackMemory   = nullptr;
        workingBuffer = nullptr;
    }
    
    __host__ void freeAllMemory() const {
        free(elementsMemory);
        free(bvhNodes);
    }

    __host__ void printInfos() const {std::cout<<"BVH : \n   Nodes : "<<nbNodes<<"\n";}
    __host__ __device__ int size() const {return nbNodes;}
    __host__ __device__ BVHNode* root() const {return &bvhNodes[0];}

    __host__ BVH* toGPU() const {
        BVHNode* bvhNodesGPU = (BVHNode*) allocGPU(sizeof(BVHNode), 2*nbPixels);
        Point3<float>* elementsMemoryGPU = (Point3<float>*) allocGPU(sizeof(Point3<float>), nbPixels);

        memCpuToGpu(bvhNodesGPU,       bvhNodes,       2*nbPixels*sizeof(BVHNode));
        memCpuToGpu(elementsMemoryGPU, elementsMemory, nbPixels*sizeof(Point3<float>));

        ArraySegment* stackMemoryGPU = nullptr;
        if(stackMemory != nullptr){
            stackMemoryGPU = (ArraySegment*) allocGPU(sizeof(ArraySegment), nbPixels);
            memCpuToGpu(stackMemoryGPU, stackMemory, nbPixels*sizeof(ArraySegment));
        }

        Point3<float>** workingBufferGPU = nullptr;
        if(workingBuffer != nullptr){
            workingBufferGPU = (Point3<float>**) allocGPU(sizeof(Point3<float>*), nbPixels);
            memCpuToGpu(workingBufferGPU, workingBuffer, nbPixels*sizeof(Point3<float>*));
        }

        BVH tmp = BVH(nbPixels, pixelSize);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        tmp.nbNodes        = nbNodes;
        tmp.bvhNodes       = bvhNodesGPU;
        tmp.elementsMemory = elementsMemoryGPU;
        tmp.stackMemory    = stackMemoryGPU;
        tmp.workingBuffer  = workingBufferGPU;
        BVH* replica = (BVH*) allocGPU(sizeof(BVH));
        checkError(cudaMemcpy(replica, &tmp, sizeof(BVH), cudaMemcpyHostToDevice));
        return replica;
    }

    __host__ void fromGPU(BVH* replica){
        BVH tmp = BVH(nbPixels, pixelSize);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        memGpuToCpu(&tmp,           replica,            sizeof(BVH));
        memGpuToCpu(bvhNodes,       tmp.bvhNodes,       2*nbPixels*sizeof(BVHNode));
        memGpuToCpu(elementsMemory, tmp.elementsMemory, nbPixels*sizeof(Point3<float>));
        
        freeGPU(tmp.bvhNodes);
        freeGPU(tmp.elementsMemory);

        if(stackMemory != nullptr){
            memGpuToCpu(stackMemory, tmp.stackMemory, nbPixels*sizeof(ArraySegment));
            freeGPU(tmp.stackMemory);
        }
        if(workingBuffer != nullptr){
            memGpuToCpu(workingBuffer, tmp.workingBuffer, nbPixels*sizeof(Point3<float>*));
            freeGPU(tmp.workingBuffer);
        }

        nbNodes   = tmp.nbNodes;
        pixelSize = tmp.pixelSize;
        freeGPU(replica);
    }

    __device__ float getLighting(const Point3<float>& origin, const Vec3<float>& invDir) const {
        uint nodeIndex = 0;
        const uint maxIndex = nbNodes;
        const float margin = pixelSize/2.0;

        while(nodeIndex < maxIndex){
            const BVHNode node = bvhNodes[nodeIndex];

            for(byte i=0; i<node.nbElements; i++){
                if(intersectBox(elementsMemory[node.elementsIndex+i], invDir, origin, margin)){
                    return 0;
                }
                // continue ??
            }

            const bool intersectRight = node.bboxRight.intersects(invDir, origin);
            const bool intersectLeft  = node.bboxLeft.intersects(invDir, origin);
            
            nodeIndex += intersectLeft + 
                (!intersectLeft && intersectRight)*(node.sizeLeft+1) + 
                (!intersectLeft && !intersectRight)*(node.sizeRight+node.sizeLeft+1);
        }
        return 1;
    }

    __device__ float getLighting2(const Point3<float>& origin, const Vec3<float>& invDir, BVHNode* const cache) const {
        /*uint nodeIndex = 0;
        const uint maxIndex = nbNodes;
        const float margin = pixelSize/TWO;

        byte nextId = 0;

        while(nodeIndex < maxIndex){

            BVHNode node;
            if(nextId < 64){
                node = cache[nextId];
                if(node.sizeLeft == 0){
                    node = cache[nextId] = bvhNodes[nodeIndex];
                }
            }else{
                node = bvhNodes[nodeIndex];
            }

            for(byte i=0; i<node.nbElements; i++){
                if(intersectBox(elementsMemory[node.elementsIndex+i], invDir, origin, margin)){
                    return 0;
                }
                // continue ??
            }

            const bool intersectRight = node.bboxRight.intersects(invDir, origin);
            const bool intersectLeft  = node.bboxLeft.intersects(invDir, origin);
            
            nodeIndex += intersectLeft + 
                (!intersectLeft && intersectRight)*(node.sizeLeft+1) + 
                (!intersectLeft && !intersectRight)*(node.sizeRight+node.sizeLeft+1);

            nextId = intersectLeft * node.cacheIdLeft + 
                (!intersectLeft && intersectRight)*(node.cacheIdRight) + 
                (!intersectLeft && !intersectRight)*255;
        }*/
        return 1;
    }


    void build(Array2D<Point3<float>*>& points) {
        const float margin = pixelSize/2;

        Bbox<float> globalBbox = Bbox<float>();
        std::vector<uint> stack = std::vector<uint>();

        uint elementsCounter = 0;
        uint nbSegments = 0;

        stack.push_back(nbSegments);
        stackMemory[nbSegments++] = ArraySegment{nullptr, points.begin(), points.end()};
        
        while(stack.size() != 0){
            ArraySegment* curSegment = &stackMemory[stack.back()];
            stack.pop_back();
            curSegment->node = new (&bvhNodes[nbNodes++]) BVHNode();

            const uint curSize = curSegment->tail - curSegment->head;
            
            if(curSize < ELEMENTS_MAX_SIZE){
                for(uint i=0; i<curSize; i++){
                    elementsMemory[elementsCounter+i] = *(curSegment->head[i]);
                }
                curSegment->node->elementsIndex = elementsCounter;
                curSegment->node->nbElements = curSize;
                elementsCounter += curSize;

            }else{
                globalBbox.setEnglobing(curSegment->head, curSize, margin);
                const uint splitIndex = split(curSegment->head, curSize, globalBbox);
                Point3<float>** middle = &(curSegment->head[splitIndex]);

                curSegment->node->bboxRight.setEnglobing(middle, curSegment->tail-middle, margin);
                stack.push_back(nbSegments);
                stackMemory[nbSegments++] = ArraySegment{curSegment, middle, curSegment->tail};
                
                curSegment->node->bboxLeft.setEnglobing(curSegment->head, middle-curSegment->head, margin);
                stack.push_back(nbSegments);
                stackMemory[nbSegments++] = ArraySegment{curSegment, curSegment->head, middle};
            }

            ArraySegment* segment = curSegment;
            while(segment->parent != nullptr){
                if(segment->node == segment->parent->node+1 ){ // If left child
                    segment->parent->node->sizeLeft++;
                }else{
                    segment->parent->node->sizeRight++;
                }
                segment = segment->parent;
            }
        }

        /*const uint nbCachedNodes = 64;
        int nodesIdFIFO[2*nbCachedNodes];
        int headFIFO = 0;
        int tailFIFO = 0;
        nodesIdFIFO[headFIFO++] = 0;
        bvhNodes[0].cacheIdLeft = 0;
        bvhNodes[0].cacheIdRight = 1;
        for(int i=0; headFIFO!=tailFIFO; i+=2){
            int curNodeId = nodesIdFIFO[tailFIFO++];
            BVHNode& node = bvhNodes[curNodeId];
            node.cacheIdLeft = i+1;
            node.cacheIdRight = i+2;
            if(i<nbCachedNodes){
                nodesIdFIFO[headFIFO++] = curNodeId+1;
                nodesIdFIFO[headFIFO++] = curNodeId+node.sizeLeft;
            }
        }*/

    }

private:
    float pixelSize; // TODO this member is always used divided by two, it can be stored divided by two
    const uint nbPixels;
    uint nbNodes = 0;

    BVHNode*        bvhNodes;
    Point3<float>*  elementsMemory;
    ArraySegment*   stackMemory;
    Point3<float>** workingBuffer;

    __host__ __device__ bool intersectBox(const Point3<float>& top, const Vec3<float>& invRayDir, const Point3<float>& rayOrigin, const float margin) const {
        float min, max;

        const float tNearX = (top.x - margin - rayOrigin.x) * invRayDir.x;
        const float tFarX  = (top.x + margin - rayOrigin.x) * invRayDir.x;

        if(tNearX > tFarX){
            min = tFarX;
            max = tNearX;
        }else{
            min = tNearX;
            max = tFarX;
        }
        
        const float tNearY = (top.y - margin - rayOrigin.y) * invRayDir.y;
        const float tFarY  = (top.y + margin - rayOrigin.y) * invRayDir.y;

        if(tNearY > tFarY){
            min = min < tFarY  ? tFarY  : min;
            max = max > tNearY ? tNearY : max;
        }else{
            min = min < tNearY ? tNearY : min;
            max = max > tFarY  ? tFarY  : max;
        }

        const float tNearZ = (-rayOrigin.z) * invRayDir.z;
        const float tFarZ  = (top.z - rayOrigin.z) * invRayDir.z;

       if(tNearZ > tFarZ){
            min = min < tFarZ  ? tFarZ  : min;
            max = max > tNearZ ? tNearZ : max;
        }else{
            min = min < tNearZ ? tNearZ : min;
            max = max > tFarZ  ? tFarZ  : max;
        }

        return min < max && min > 0;
    }

    __host__ __device__ bool intersectSphere(const Point3<float>& top, const Ray<float>& ray, const float radius) const {
        ray.getDirection().normalize();
        const Point3<float> center = Point3<float>(top.x, top.y, top.z-pixelSize/2);
        const float radius_squared = radius*radius;
        const Vec3<float> d_co = ray.getOrigin() - center;
        const float d_co_norm_sqr = d_co.getNormSquared();
        if(d_co_norm_sqr <= radius_squared) return false;
        const float tmp = ray.getDirection().dot(d_co);
        const float delta = tmp*tmp - (d_co_norm_sqr - radius_squared);
        const float t = -tmp-sqrt(delta);
        return delta >= 0 && t > 0;
    }

    __host__ __device__ int split(Point3<float>** points, const uint size, const Bbox<float>& bbox) const {
        const float dx = bbox.getEdgeLength('X');
        const float dy = bbox.getEdgeLength('Y');
        //const float dz = bbox->getEdgeLength('Z');
        const Point3<float> center = bbox.getCenter();

        int nbLeft  = 0;
        int nbRight = 0;

        for(uint i=0; i<size; i++){
            Point3<float>* const point = points[i];
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
        for(uint i=0; i<size; i++){
            points[i] = workingBuffer[i];
        }
        return nbLeft;
    }
};