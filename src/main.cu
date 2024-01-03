#include <cstddef>
#include <iostream>
#include <cmath>

#include <chrono>
#include <atomic>
#include <ostream>

#include "primitives/Vec3.cuh"
#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "array/Array.cuh"
#include "primitives/Point3.cuh"
#include "primitives/Ray.cuh"

#define PI 3.14159265358979323846

__global__ 
void initRender(int maxX, int maxY, curandState* randomState) {
   const int x = threadIdx.x + blockIdx.x * blockDim.x;
   const int y = threadIdx.y + blockIdx.y * blockDim.y;
   if(x>=maxX || y>=maxY) return;
   const int index = y*maxX + x;
   curand_init(1423, index, 0, &randomState[index]);
}


__global__
void trace(float* data, Point3<float>* points, int maxX, int maxY, BVH<float>* bvh, const int raysPerPoint, curandState* randomState, BVHNode<float>** traceBuffer, int traceBufferSize){
    /*const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=maxX || y>=maxY) return;
    const int index = y*maxX + x;

    curandState localRndState = randomState[index];

    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray = Ray<float>(points[index], direction);

    float result = 0;
    for(int i=0; i<raysPerPoint; i++){
        ray.getDirection().setRandomInHemisphere(localRndState, i%4);
        result += bvh->getLighting(ray, &traceBuffer[index*traceBufferSize]);
    }
    data[index] = result/raysPerPoint;*/
}

int main(){
    const bool USE_GPU = false;
    const bool PRINT_INFOS = true;
    const char* filename = "data/input.tif";
    const char* outputFilename = "data/output.tif";

    Raster raster = Raster(filename, outputFilename);

    if(PRINT_INFOS){
        raster.printInfos();
        printDevicesInfos();     
    }

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
            points[index] = Point3<float>((float)x/2,(float)y/2,data[index]);
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

    cudaFree(workingBuffer);
    cudaFree(workingBufferPMemory);
    cudaFree(stackMemory);
    
    std::cout << "BVH built\n";

    // Trace
    constexpr unsigned int RAYS_PER_POINT = 512;

    BVHNode<float>** traceBuffer;
    const int traceBufferSizePerThread = std::log2(bvh->size())+1;
    checkError(cudaMallocManaged(&traceBuffer, nbPixels*traceBufferSizePerThread*sizeof(BVHNode<float>*)));

    if(USE_GPU){
        const dim3 threads(8,8);
        const dim3 blocks(raster.getWidth()/threads.x+1, raster.getHeight()/threads.y+1);
        
        curandState* randomState;
        checkError(cudaMallocManaged((void **)& randomState, nbPixels*sizeof(curandState)));
        
        std::cout << "Initializing tracing...\n";

        initRender<<<blocks, threads>>>(raster.getWidth(), raster.getHeight(), randomState);
        checkError(cudaGetLastError());
        checkError(cudaDeviceSynchronize());

        std::cout << "Start tracing...\n";

        trace<<<blocks, threads>>>(
            data, points, raster.getWidth(), raster.getHeight(), 
            bvh, RAYS_PER_POINT, randomState, traceBuffer, traceBufferSizePerThread);
        checkError(cudaGetLastError());
        checkError(cudaDeviceSynchronize());

    }else{
        std::cout << "Start tracing...\n";

        constexpr int NB_STRATIFIED_DIRS = 32;

        float progress = 0;
        float nextProgress = 0.1;

        #pragma omp parallel for
        for(int y=0; y<raster.getHeight(); y++){
            for(int x=0; x<raster.getWidth(); x++){
                const int index = y*raster.getWidth() + x;

                Point3<float> origin  = points[index];
                Vec3<float> direction = Vec3<float>(0,0,0);
                Ray<float> ray        = Ray<float>(origin, direction);

                float result = 0;
                for(int i=0; i<RAYS_PER_POINT; i++){
                    float p = ray.getDirection().setRandomInHemisphereImportance2( NB_STRATIFIED_DIRS , i%NB_STRATIFIED_DIRS );
                    result += bvh->getLighting(ray, &traceBuffer[index*traceBufferSizePerThread])/p;
                }
                data[index] = result/RAYS_PER_POINT;
            }

            #pragma omp atomic
            progress++;

            if( ((float)progress)/raster.getHeight() >= nextProgress){
                std::cout << "Progress " << 100*nextProgress << "%\n";
                std::flush(std::cout);
                nextProgress += 0.1;
            }
        }
    }

    std::cout << "Tracing finished...\n";

    raster.writeData(data);

    cudaFree(bvh);
    cudaFree(elementsMemory);
    cudaFree(bboxMemory);
    cudaFree(BVHNodeMemory);

    cudaFree(data);


    std::cout << "Finished \n";
    return 0;
}