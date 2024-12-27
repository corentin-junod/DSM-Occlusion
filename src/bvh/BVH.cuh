#pragma once

#include "../primitives/Bbox.cuh"
#include "../primitives/Vec3.cuh"
#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <cstdio>
#include <iostream>
#include <ostream>

constexpr byte ELEMENTS_MAX_SIZE = 1;

struct __align__(16) BVHNode{
    Bbox<float> bboxLeft  = Bbox<float>();
    Bbox<float> bboxRight = Bbox<float>();
    uint sizeLeft  = 0;
    uint sizeRight = 0;
    bool isLeafe = false;
};

struct ArraySegment{
    ArraySegment* parent;
    Point3<float>** head;
    Point3<float>** tail;
    BVHNode* node = nullptr;
};

class BVH{
public:
    __host__ BVH(const uint nbPixels, const float pixelSize): nbPixels(nbPixels), margin(pixelSize/2) {
        bvhNodes       = (BVHNode*)        calloc(2*nbPixels, sizeof(BVHNode));
        stackMemory    = (ArraySegment*)   calloc(2*nbPixels, sizeof(ArraySegment));
        workingBuffer  = (Point3<float>**) calloc(nbPixels,   sizeof(Point3<float>*));
    }

    __host__ void freeAfterBuild() {
        free(stackMemory);
        free(workingBuffer);
        stackMemory   = nullptr;
        workingBuffer = nullptr;
    }
    
    __host__ void freeAllMemory() const {
        free(bvhNodes);
    }

    __host__ void printInfos() const {std::cout<<"BVH : \n   Nodes : "<<nbNodes<<"\n";}
    __host__ __device__ int size() const {return nbNodes;}
    __host__ __device__ BVHNode* root() const {return &bvhNodes[0];}

    __host__ BVH* toGPU() const {
        BVHNode* bvhNodesGPU   = (BVHNode*) allocGPU(sizeof(BVHNode), 2*nbPixels);

        memCpuToGpu(bvhNodesGPU, bvhNodes, 2*nbPixels*sizeof(BVHNode));

        ArraySegment* stackMemoryGPU = nullptr;
        if(stackMemory != nullptr){
            stackMemoryGPU = (ArraySegment*) allocGPU(sizeof(ArraySegment), 2*nbPixels);
            memCpuToGpu(stackMemoryGPU, stackMemory, 2*nbPixels*sizeof(ArraySegment));
        }

        Point3<float>** workingBufferGPU = nullptr;
        if(workingBuffer != nullptr){
            workingBufferGPU = (Point3<float>**) allocGPU(sizeof(Point3<float>*), nbPixels);
            memCpuToGpu(workingBufferGPU, workingBuffer, nbPixels*sizeof(Point3<float>*));
        }

        BVH tmp = BVH(nbPixels, margin*2);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        tmp.nbNodes       = nbNodes;
        tmp.bvhNodes      = bvhNodesGPU;
        tmp.stackMemory   = stackMemoryGPU;
        tmp.workingBuffer = workingBufferGPU;
        BVH* replica = (BVH*) allocGPU(sizeof(BVH));
        checkError(cudaMemcpy(replica, &tmp, sizeof(BVH), cudaMemcpyHostToDevice));
        return replica;
    }

    __host__ void fromGPU(BVH* replica){
        BVH tmp = BVH(nbPixels, margin*2);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        memGpuToCpu(&tmp,     replica,      sizeof(BVH));
        memGpuToCpu(bvhNodes, tmp.bvhNodes, 2*nbPixels*sizeof(BVHNode));
        
        freeGPU(tmp.bvhNodes);

        if(stackMemory != nullptr){
            memGpuToCpu(stackMemory, tmp.stackMemory, 2*nbPixels*sizeof(ArraySegment));
            freeGPU(tmp.stackMemory);
        }
        if(workingBuffer != nullptr){
            memGpuToCpu(workingBuffer, tmp.workingBuffer, nbPixels*sizeof(Point3<float>*));
            freeGPU(tmp.workingBuffer);
        }

        nbNodes = tmp.nbNodes;
        margin = tmp.margin;
        freeGPU(replica);
    }

    __device__ float getLighting(const Point3<float>& origin, Vec3<float>& dir, curandState& localRndState, const uint maxBounces=0) const {
        const uint maxIndex = nbNodes;
        Vec3<float> invDir(fdividef(1,dir.x), fdividef(1,dir.y), fdividef(1,dir.z));
        Point3<float> curOrigin = Point3<float>(origin.x, origin.y, origin.z);
        uint nodeIndex = 0;
        uint bounces = 0;
        bool wasHit = false;
        float radiance = 1;
        const float reflectance = 1;

        while(nodeIndex < maxIndex){
            const BVHNode node = bvhNodes[nodeIndex];
            if(node.isLeafe && wasHit){
                if(bounces == maxBounces){
                    return 0;
                }
                bounces++;
                nodeIndex = 0;
                wasHit = false;
                node.bboxLeft.bounceRay(dir, curOrigin, localRndState);
                // For a diffuse BSDF, with R the reflectance, the BSDF = R/PI. 
                // This BSDF must then be multiplied by cos(theta) according to the rendering equation.
                // The PDF of a cosine-weighted random direction in the hemisphere is cos(theta)/PI
                // We have : (R/PI) * cos(theta) / (cos(theta)/PI) = R, hence the multiplication by only R
                radiance *= reflectance;
                invDir = Vec3<float>(fdividef(1,dir.x), fdividef(1,dir.y), fdividef(1,dir.z));
            }else if(node.bboxLeft.intersects(invDir, curOrigin)){
                nodeIndex += 1;
                wasHit=true;
            }else if(node.bboxRight.intersects(invDir, curOrigin)){
                nodeIndex += node.sizeLeft+1;
                wasHit=true;
            }else{
                nodeIndex += node.sizeRight+node.sizeLeft+1;
                wasHit=false;
            }
        }
        return radiance;
    }

    void build(Array2D<Point3<float>*>& points) {
        Bbox<float> globalBbox = Bbox<float>();

        uint* stack = (uint*) calloc(2*nbPixels, sizeof(uint));
        uint stackSize=0;

        uint elementsCounter = 0;
        uint nbSegments = 0;

        stack[stackSize++] = nbSegments;
        stackMemory[nbSegments++] = ArraySegment{nullptr, points.begin(), points.end()};

        TIMED_INIT(split_time)
        TIMED_INIT(set_englobing_time)
        TIMED_INIT(update_segment_size_time)

        while(stackSize != 0){
            ArraySegment* const curSegment = &stackMemory[stack[--stackSize]];
            curSegment->node = new (&bvhNodes[nbNodes++]) BVHNode();

            const uint curSize = curSegment->tail - curSegment->head;
            
            if(curSize <= ELEMENTS_MAX_SIZE){
                curSegment->node->isLeafe = true;
                curSegment->node->bboxLeft.setEnglobing(curSegment->head, curSize, margin);
                elementsCounter += curSize;

            }else{
                TIMED_ACC(set_englobing_time, globalBbox.setEnglobing(curSegment->head, curSize, margin); )
                TIMED_ACC(split_time, const uint splitIndex = split(curSegment->head, curSize, globalBbox); )

                Point3<float>** const middle = &(curSegment->head[splitIndex]);

                TIMED_ACC(set_englobing_time,
                    curSegment->node->bboxRight.setEnglobing(middle, curSegment->tail-middle, margin);
                    stack[stackSize++] = nbSegments;
                    stackMemory[nbSegments++] = ArraySegment{curSegment, middle, curSegment->tail};

                    curSegment->node->bboxLeft.setEnglobing(curSegment->head, middle-curSegment->head, margin);
                    stack[stackSize++] = nbSegments;
                    stackMemory[nbSegments++] = ArraySegment{curSegment, curSegment->head, middle};
                )
            }

            TIMED_ACC(update_segment_size_time,
                ArraySegment* segment = curSegment;
                while(segment->parent != nullptr){
                    if(segment->node == segment->parent->node+1){ // If left child
                        segment->parent->node->sizeLeft++;
                    }else{
                        segment->parent->node->sizeRight++;
                    }
                    segment = segment->parent;
                }
            )
        }

        TIMED_PRINT(split_time)
        TIMED_PRINT(set_englobing_time)
        TIMED_PRINT(update_segment_size_time)

        free(stack);
    }

private:
    float margin;
    const uint nbPixels;
    uint nbNodes = 0;

    BVHNode*        bvhNodes;
    ArraySegment*   stackMemory;
    Point3<float>** workingBuffer;

    __host__ int split(Point3<float>** const points, const uint size, const Bbox<float>& bbox) const {
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