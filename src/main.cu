#include <cstddef>
#include <iostream>
#include <cmath>

#include <chrono>

#include "primitives/Vec3.cuh"
#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "array/Array.cuh"
#include "primitives/Point3.cuh"
#include "primitives/Ray.cuh"

/*
__global__  
void buildBVH(
        BVH<float>** bvh, Array<Point3<float>>& pointsArray, ArraySegment<float>* stackMemory, 
        Point3<float>** workingBufferPMemory, BVHNode<float>* BVHNodeMemory, Bbox<float>* bboxMemory, Array<Point3<float>>* elementsMemory){
    *bvh = new BVH<float>(pointsArray, stackMemory, workingBufferPMemory, BVHNodeMemory, bboxMemory, elementsMemory);
}
*/

__global__ 
void initRender(int maxX, int maxY, curandState* randomState) {
   const int x = threadIdx.x + blockIdx.x * blockDim.x;
   const int y = threadIdx.y + blockIdx.y * blockDim.y;
   if(x>=maxX || y>=maxY) return;
   const int index = y*maxX + x;
   curand_init(1423, index, 0, &randomState[index]);
}


__global__
void trace(float* data, Point3<float>* points, int maxX, int maxY, BVH<float>* bvh, int raysPerPoint, curandState* randomState, BVHNode<float>** traceBuffer, int traceBufferSize){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=maxX || y>=maxY) return;
    const int index = y*maxX + x;

    curandState localRndState = randomState[index];

    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);

    Ray<float>* rayMem;
    cudaMalloc(&rayMem, bvh->size()*sizeof(Ray<float>));
    Ray<float>* ray = new(rayMem) Ray<float>(origin, direction);

    float result = 0;
    for(int i=0; i<raysPerPoint; i++){
        ray->setDirection(Vec3<float>::randomInHemisphere(localRndState));
        result += bvh->isIntersecting(*ray, &traceBuffer[index*traceBufferSize])?0.0:1.0;
    }
    data[index] = result/raysPerPoint;
}

int main(){
    const char* filename = "data/input.tif";
    const char* outputFilename = "data/output.tif";
    Raster raster = Raster(filename, outputFilename);

    raster.printInfos();
    //printDevicesInfos();

    const unsigned int nbPixels = raster.getWidth()*raster.getHeight();

    // Read data from raster
    float* data;
    checkError(cudaMallocManaged(&data, nbPixels*sizeof(float)));
    raster.readData(data);

    // Create points in 3D
    Point3<float>* points;
    checkError(cudaMallocManaged(&points, nbPixels*sizeof(Point3<float>)));

    Point3<float>** pointsArrayContent;
    checkError(cudaMallocManaged(&pointsArrayContent, nbPixels*sizeof(Point3<float>*)));

    Point3<float>* pointsArrayMemory;
    checkError(cudaMallocManaged(&pointsArrayMemory, sizeof(Array<Point3<float>*>)));
    Array<Point3<float>*>* pointsArray = new (pointsArrayMemory) Array<Point3<float>*>(pointsArrayContent, nbPixels);
    

    for(int y=0; y<raster.getHeight(); y++){
        for(int x=0; x<raster.getWidth(); x++){
            const int index = y*raster.getWidth()+x;
            points[index] = Point3<float>(x,y,data[index]);
            (*pointsArray)[index] = &(points[index]);
        }
    }

    // Build BVH
    std::cout << "Building BVH...\n";

    ArraySegment<float>* stackMemory;
    checkError(cudaMallocManaged(&stackMemory, nbPixels*sizeof(ArraySegment<float>)));

    Point3<float>** workingBufferPMemory;
    checkError(cudaMallocManaged(&workingBufferPMemory, nbPixels*sizeof(Point3<float>*)));

    BVHNode<float>* BVHNodeMemory;
    checkError(cudaMallocManaged(&BVHNodeMemory, 2*nbPixels*sizeof(BVHNode<float>)));

    Bbox<float>* bboxMemory;
    checkError(cudaMallocManaged(&bboxMemory, (int)2*nbPixels*sizeof(Bbox<float>)));

    Array<Point3<float>>* elementsMemory;
    checkError(cudaMallocManaged(&elementsMemory, (int)2*nbPixels*sizeof(Array<Point3<float>>)));


    Array<Point3<float>*>* workingBuffer;
    checkError(cudaMallocManaged(&workingBuffer, sizeof(Array<Point3<float>*>)));
    workingBuffer = new (workingBuffer) Array<Point3<float>*>(workingBufferPMemory, pointsArray->size());

    BVH<float>* bvh;
    checkError(cudaMallocManaged(&bvh, sizeof(BVH<float>)));
    bvh = new (bvh) BVH<float>(*pointsArray, stackMemory, *workingBuffer, BVHNodeMemory, bboxMemory, elementsMemory);

    //BVH<float>* bvh;
    //buildBVH<<<1,1>>>(&bvh, *pointsArray, stackMemory, workingBufferPMemory, BVHNodeMemory, bboxMemory, elementsMemory);
    
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());
    std::cout << "BVH built\n";

    // Trace
    constexpr unsigned int RAYS_PER_POINT = 15;

    std::cout << "Start tracing...\n";


    const dim3 threads(8,8);
    const dim3 blocks(raster.getWidth()/threads.x+1, raster.getHeight()/threads.y+1);

    BVHNode<float>** traceBuffer;
    checkError(cudaMallocManaged(&traceBuffer, nbPixels*std::log2(bvh->size())*sizeof(BVHNode<float>*)));
    
    curandState* randomState;
    checkError(cudaMallocManaged((void **)& randomState, nbPixels*sizeof(curandState)));
    
    initRender<<<blocks, threads>>>(raster.getWidth(), raster.getHeight(), randomState);
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    trace<<<blocks, threads>>>(
        data, points, raster.getWidth(), raster.getHeight(), 
        bvh, RAYS_PER_POINT, randomState, traceBuffer, std::log2(bvh->size()));
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());


    /*BVHNode<float>** traceBuffer;
    checkError(cudaMallocManaged(&traceBuffer, bvh->size()*sizeof(BVHNode<float>*)));

    for(int y=0; y<raster.getHeight(); y++){
        for(int x=0; x<raster.getWidth(); x++){
            const int index = y*raster.getWidth() + x;

            Point3<float> origin  = points[index];
            Vec3<float> direction = Vec3<float>(0,0,0);
            Ray<float> ray        = Ray<float>(origin, direction);
            float result = 0;
            for(int i=0; i<RAYS_PER_POINT; i++){
                ray.setDirection(Vec3<float>::randomInHemisphere());
                result += bvh->isIntersecting(ray, traceBuffer)?0.0:1.0;
            }
            data[index] = result/RAYS_PER_POINT;
        }
    }*/


    std::cout << "Tracing finished...\n";

    raster.writeData(data);

    checkError(cudaFree(data));

    std::cout << "Finished \n";
    return 0;
}