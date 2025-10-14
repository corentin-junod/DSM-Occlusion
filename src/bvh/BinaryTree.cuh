#pragma once

#include "../primitives/Bbox.cuh"
#include "../primitives/Vec3.cuh"
#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <cstdio>
#include <iostream>
#include <ostream>
#include <string>

struct __align__(16) BinaryTreeNode{
    float value = 0;
    bool isLeafe = false;
};

class BinaryTree{
public:
    __host__ BinaryTree(const uint nbPixels, const float pixelSize): nbPixels(nbPixels), margin(pixelSize/2) {
        btNodes = (BinaryTreeNode*) calloc(2*nbPixels, sizeof(BinaryTreeNode));
        nbNodes = 2*nbPixels;
    }

    __host__ void freeAfterBuild() {}
    
    __host__ void freeAllMemory() const {
        free(btNodes);
    }

    __host__ void printInfos() const {std::cout<<"Binary Tree : \n   Nodes : "<<nbNodes<<"\n";}
    __host__ __device__ int size() const {return nbNodes;}
    __host__ __device__ BinaryTreeNode* root() const {return &btNodes[0];}

    __host__ BinaryTree* toGPU() const {
        size_t nbMaxNodes = 2*nbPixels;
        BinaryTreeNode* btNodesGPU = (BinaryTreeNode*) allocGPU(sizeof(BinaryTreeNode), nbMaxNodes);

        memCpuToGpu(btNodesGPU, btNodes, nbMaxNodes*sizeof(BinaryTreeNode));

        BinaryTree tmp = BinaryTree(nbPixels, margin*2);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        tmp.nbNodes = nbNodes;
        tmp.btNodes = btNodesGPU;
        tmp.rasterHeight = rasterHeight;
        tmp.rasterWidth = rasterWidth;
        BinaryTree* replica = (BinaryTree*) allocGPU(sizeof(BinaryTree));
        checkError(cudaMemcpy(replica, &tmp, sizeof(BinaryTree), cudaMemcpyHostToDevice));
        return replica;
    }

    __host__ void fromGPU(BinaryTree* replica){
        size_t nbMaxNodes = 2*nbPixels;
        BinaryTree tmp = BinaryTree(nbPixels, margin*2);
        tmp.freeAfterBuild();
        tmp.freeAllMemory();
        memGpuToCpu(&tmp,     replica,      sizeof(BinaryTree));
        memGpuToCpu(btNodes, tmp.btNodes, nbMaxNodes*sizeof(BinaryTreeNode));
        
        freeGPU(tmp.btNodes);

        nbNodes = tmp.nbNodes;
        margin = tmp.margin;
        rasterHeight = tmp.rasterHeight;
        rasterWidth = tmp.rasterWidth;
        freeGPU(replica);
    }

    __device__ float getLighting(const Point3<float>& origin, Vec3<float>& dir, curandState& localRndState, const LightingParams& params) const {
        const uint maxIndex = nbNodes;
        Vec3<float> invDir(fdividef(1,dir.x), fdividef(1,dir.y), fdividef(1,dir.z));
        Point3<float> curOrigin = Point3<float>(origin.x, origin.y, origin.z);
        uint nodeIndex = 0;
        uint bounces = 0;
        bool wasHit = false;
        float radianceFactor = 1;

        
        float pixelSize = 0.1;
        /*curOrigin.x /= pixelSize;
        curOrigin.y /= pixelSize;
        curOrigin.z = curOrigin.z / pixelSize + 0.1;*/

        uint minX = 0;
        uint maxX = rasterWidth;
        uint minY = 0;
        uint maxY = rasterHeight;
        uint depth = 0;

        constexpr uint stacksize = 2048;
        uint stack[stacksize];
        uint stackMinX[stacksize];
        uint stackMaxX[stacksize];
        uint stackMinY[stacksize];
        uint stackMaxY[stacksize];
        uint stackDepth[stacksize];
        uint stackPointer = 0;

        Bbox<float> bboxLeft;
        Bbox<float> bboxRight;


        while(nodeIndex < maxIndex){
            const BinaryTreeNode node = btNodes[nodeIndex];

            if(/*node.isLeafe*/ depth == 18){
                if(bounces == params.maxBounces){
                    //return node.value;
                    return params.ambientPower;
                }
                // TODO Handle bounces
                /*bounces++;
                nodeIndex = 0;
                wasHit = false;
                node.bboxLeft.bounceRay(dir, curOrigin, localRndState);
                // For a diffuse BSDF, with R the reflectance, the BSDF = R/PI. 
                // This BSDF must then be multiplied by cos(theta) according to the rendering equation.
                // The PDF of a cosine-weighted random direction in the hemisphere is cos(theta)/PI
                // We have : (R/PI) * cos(theta) / (cos(theta)/PI) = R, hence the multiplication by only R
                radianceFactor *= params.materialReflectance;
                invDir = Vec3<float>(fdividef(1,dir.x), fdividef(1,dir.y), fdividef(1,dir.z));*/
            }

            uint leftChildIndex;
            uint rightChildIndex;
            
            if(depth%2 == 0){
                uint firstPyramidIndex = (1 << depth) -1; // 255
                uint firstNextPyramidIndex = (1 << (depth+1))-1; // 511
                uint lineLength = sqrtf(firstPyramidIndex+1); // 16 TODO prendre la racine carré d'une puissance de deux plutôt que sqrt
                uint lineNumber = 2*((nodeIndex - firstPyramidIndex) / lineLength); // 2
                uint lineOffset = (nodeIndex - firstPyramidIndex) % lineLength; // 1
                leftChildIndex = firstNextPyramidIndex + lineNumber*lineLength + lineOffset; // 528 <--> 544
                rightChildIndex = leftChildIndex + lineLength;

                bboxLeft = Bbox<float>(pixelSize*minX-pixelSize/2.0f, pixelSize*maxX-pixelSize/2.0f, pixelSize*minY-pixelSize/2.0f, pixelSize*(maxY+minY)/2-pixelSize/2.0f, btNodes[leftChildIndex].value);
                bboxRight = Bbox<float>(pixelSize*minX-pixelSize/2.0f, pixelSize*maxX-pixelSize/2.0f, pixelSize*(maxY+minY)/2-pixelSize/2.0f, pixelSize*maxY-pixelSize/2.0f, btNodes[rightChildIndex].value);
            }else{
                leftChildIndex = (nodeIndex + 1) * 2 - 1;
                rightChildIndex = leftChildIndex+1;

                bboxLeft = Bbox<float>(pixelSize*minX-pixelSize/2.0f, pixelSize*(maxX+minX)/2-pixelSize/2.0f, pixelSize*minY-pixelSize/2.0f, pixelSize*maxY-pixelSize/2.0f, btNodes[leftChildIndex].value);
                bboxRight = Bbox<float>(pixelSize*(maxX+minX)/2-pixelSize/2.0f, pixelSize*maxX-pixelSize/2.0f, pixelSize*minY-pixelSize/2.0f, pixelSize*maxY-pixelSize/2.0f, btNodes[rightChildIndex].value);
            }

            /*if(depth%2 == 0){
                if(curOrigin.y < (maxY+minY)/2.0f ){
                    nodeIndex = leftChildIndex;
                    maxY = (maxY+minY)/2.0f;
                }
                else {
                    nodeIndex = rightChildIndex;
                    minY = (maxY+minY)/2.0f;
                }
            }else{
                if(curOrigin.x < (maxX+minX)/2.0f ){
                    nodeIndex = leftChildIndex;
                    maxX = (maxX+minX)/2.0f;
                }
                else {
                    nodeIndex = rightChildIndex;
                    minX = (maxX+minX)/2.0f;
                }
            }
            wasHit=true;
            depth++;
            continue;*/
            

            wasHit=false;
            uint originalMaxX = maxX;
            uint originalMaxY = maxY;
            uint originalDepth = depth; 

            if(bboxLeft.intersects(invDir, curOrigin)){
                //if(depth < 17 || (curOrigin.x < bboxLeft.minX|| curOrigin.x > bboxLeft.maxX || curOrigin.y < bboxLeft.minY || curOrigin.y > bboxLeft.maxY ))
                //{
                    if(depth%2 == 0){
                        maxY = (maxY+minY)/2;
                    }else{
                        maxX = (maxX+minX)/2;
                    }
                    nodeIndex = leftChildIndex;
                    wasHit=true;
                    depth++;
                //}
              
            }

            if(bboxRight.intersects(invDir, curOrigin)){
                //if(depth < 17 || (curOrigin.x < bboxRight.minX|| curOrigin.x > bboxRight.maxX || curOrigin.y < bboxRight.minY || curOrigin.y > bboxRight.maxY ))
                //{
                    if(wasHit){
                        stack[stackPointer] = rightChildIndex;
                        stackMinX[stackPointer] = originalDepth%2 == 0 ? minX : (originalMaxX+minX)/2;
                        stackMaxX[stackPointer] = originalMaxX;
                        stackMinY[stackPointer] = originalDepth%2 == 0 ? (originalMaxY+minY)/2 : minY;
                        stackMaxY[stackPointer] = originalMaxY;
                        stackDepth[stackPointer] = originalDepth+1;
                        stackPointer++;
                        
                    }else{
                        if(depth%2 == 0){
                            minY = (originalMaxY+minY)/2;
                        }else{
                            minX = (originalMaxX+minX)/2;
                        }
                        nodeIndex = rightChildIndex;
                        wasHit=true;
                        depth++;
                    }
                //}
            }

            if(!wasHit){
                if(stackPointer > 0){
                    stackPointer--;
                    nodeIndex = stack[stackPointer];
                    minX = stackMinX[stackPointer];
                    maxX = stackMaxX[stackPointer];
                    minY = stackMinY[stackPointer];
                    maxY = stackMaxY[stackPointer];
                    depth = stackDepth[stackPointer];
                } else {
                    break; 
                }
            }

        }
        return 1;
        float radiance = 0;

        dir.normalize();

        const float thetaSun = (90 - params.sunElevation) * PI/180;
        const float phiSun = params.sunAzimuth * PI/180;
        const float thetaDir = acosf(dir.z);
        const float phiDir = atan2f(dir.y, dir.x);
        const float diffPhi = fabsf(phiDir+PI - phiSun+PI); // +PI so that we are always with a positive difference
        const float centralAngle = acosf(cosf(thetaSun)*cosf(thetaDir) + sinf(thetaSun)*sinf(thetaDir)*cosf(diffPhi));

        if(centralAngle < (params.sunAngularDiam*PI/180)/2.0){
            radiance += radianceFactor * params.sunPower;
        }

        radiance += radianceFactor * params.skyPower;
        return radiance;
    }

    void build(Array2D<Point3<float>*>& points) {
    // Note : we need points to be a square array with a side length that is a power of 2

        rasterHeight = points.height();
        rasterWidth = points.width();

        std::cout << "Tile size : " << std::to_string(rasterWidth) << " x " << std::to_string(rasterHeight) << "\n";
        std::cout << "Nb points : " << std::to_string(points.size()) << "\n";

        // Write all the leaves
        for(int i=0; i<points.size(); i++){
            btNodes[nbPixels-1 + i].value = points.at(i%rasterHeight, i/rasterHeight)->z;
            btNodes[nbPixels-1 + i].isLeafe = true;
        }

        // Write all pyramids
        uint remainingLength = points.size();
        uint curLineStart;
        uint curHeight = rasterHeight;
        uint curWidth = rasterWidth/2;
        bool switcher = true;

        while(remainingLength > 1){
            remainingLength /= 2;
            curLineStart = remainingLength - 1;

            std::cout << "Writing tree line of size " << std::to_string(remainingLength) << " at " << std::to_string(curLineStart )<< " width : " << std::to_string(curWidth);
            
            if(switcher){

                for(int i=0; i<remainingLength; i++){
                    uint curIndex = curLineStart + i;
                    uint childIndex1 = (curIndex+1)*2 -1;
                    uint childIndex2 = childIndex1 + 1;
                    btNodes[curIndex].value = std::max(btNodes[childIndex1].value, btNodes[childIndex2].value);
                }
                curHeight /= 2;

            }else{

                for(int i=0; i<remainingLength; i++){
                    uint curIndex = curLineStart + i;
                    uint lineNumber = i/curWidth;
                    uint lineOffset = i%curWidth;
                    uint childIndex1 = curLineStart*2+1 + 2*lineNumber*curWidth + lineOffset;
                    uint childIndex2 = childIndex1 + curWidth;
                    btNodes[curIndex].value = std::max(btNodes[childIndex1].value, btNodes[childIndex2].value);
                }
                curWidth /= 2;

            }
            switcher = !switcher;

            std::cout << " for pyramid of size " << std::to_string(curWidth) << " x " << std::to_string(curHeight) << "\n";
        }
    }

private:
    uint rasterWidth;
    uint rasterHeight;
    float margin;
    const uint nbPixels;
    uint nbNodes = 0;
    BinaryTreeNode* btNodes;
};