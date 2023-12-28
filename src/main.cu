#include <cstddef>
#include <iostream>

#include <chrono>

#include "primitives/Vec3.cuh"
#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "array/Array.cuh"
#include "primitives/Point3.cuh"
#include "primitives/Ray.cuh"

__global__  
void buildBVH(BVH<float>** bvh, Array<Point3<float>>& pointsArray){
    *bvh = new BVH<float>(pointsArray);
}

__global__ 
void initRender(int maxX, int maxY, curandState* randomState) {
   const int x = threadIdx.x + blockIdx.x * blockDim.x;
   const int y = threadIdx.y + blockIdx.y * blockDim.y;
   if(x>=maxX || y>=maxY) return;
   const int index = y*maxX + x;
   curand_init(1423, index, 0, &randomState[index]);
}

__global__
void trace(float* data, Array<Point3<float>>& pointsArray, int maxX, int maxY, BVH<float>* bvh, int rayPerPoint, curandState* randomState){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=maxX || y>=maxY) return;
    const int index = y*maxX + x;

    Point3<float> origin = Point3<float>(0,0,0);
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray = Ray<float>(origin, direction);

    BVHNode<float>* buffer;
    cudaMalloc(&buffer, bvh->size()*sizeof(BVHNode<float>));

    ray.setOrigin(pointsArray[index]);
    float result = 0;
    for(int i=0; i<rayPerPoint; i++){
        ray.setDirection(Vec3<float>::randomInHemisphere(randomState));
        result += bvh->isIntersectingIterGPU(ray, &buffer)?0.0:1.0;
    }
    data[index] = result/rayPerPoint;
}

int main(){
    const char* filename = "data/data.tif";
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

    void* pointsArrayLoc;
    checkError(cudaMallocManaged(&pointsArrayLoc, sizeof(Array<Point3<float>>)));
    Array<Point3<float>>* pointsArray = new(pointsArrayLoc) Array<Point3<float>>(points, nbPixels);
    
    /*
    Array<Point3<float>>* memLoc;
    checkError(cudaMallocManaged(&memLoc, sizeof(Array<Point3<float>>)));
    Array<Point3<float>>* pointsArray = new(memLoc) Array<Point3<float>>(points, nbPixels);
    */

    for(int y=0; y<raster.getHeight(); y++){
        for(int x=0; x<raster.getWidth(); x++){
            const int index = y*raster.getWidth()+x;
            (*pointsArray)[index] = Point3<float>(x,y,data[index]);
        }
    }

    // Build BVH
    std::cout << "Building BVH...\n";
    BVH<float>* bvh;
    buildBVH<<<1,1>>>(&bvh, *pointsArray);
    std::cout << "BVH built\n";

    // Trace
    constexpr unsigned int NB_RAY_PER_POINT = 10;

    std::cout << "Start tracing...\n";
    //curandState randState = curandState();
    //curand_init(4132, 0, 0, &randState);

    //auto t1 = std::chrono::high_resolution_clock::now();

    /*std::vector<BVHNode<float>*> toCheck = std::vector<BVHNode<float>*>();
    toCheck.reserve(bvh.size());

    Point3<float> origin = Point3<float>(0,0,0);
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray = Ray<float>(origin, direction);
    for(std::size_t i=0; i<pointsArray.size(); i++){
        ray.setOrigin(pointsArray[i]);
        float result = 0;
        for(std::size_t j=0; j<NB_RAY_PER_POINT; j++){
            ray.setDirection(Vec3<float>::randomInHemisphere());
            result += bvh.isIntersectingIter(ray, toCheck)?0.0:1.0;
        }
        data[i] = result/NB_RAY_PER_POINT;
    }*/

    //auto t2 = std::chrono::high_resolution_clock::now();

/*
    Point3<float> origin = Point3<float>(0,0,0);
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray2 = Ray<float>(origin, direction);
    for(std::size_t i=0; i<pointsArray.size(); i++){
        ray2.setOrigin(pointsArray[i]);
        float result = 0;
        for(std::size_t j=0; j<NB_RAY_PER_POINT; j++){
            ray2.setDirection(Vec3<float>::randomInHemisphere());
            result += bvh.isIntersectingRec(ray2)?0.0:1.0;
        }
        data[i] = result/NB_RAY_PER_POINT;
    }
*/

    const dim3 threads(8,8);
    const dim3 blocks(raster.getWidth()/threads.x+1, raster.getHeight()/threads.y+1);
    curandState* randomState;
    checkError(cudaMalloc((void **)& randomState, nbPixels*sizeof(curandState)));
    initRender<<<blocks, threads>>>(raster.getWidth(), raster.getHeight(), randomState);
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());
    trace<<<blocks, threads>>>(
        data, *pointsArray, raster.getWidth(), raster.getHeight(), bvh, NB_RAY_PER_POINT, randomState);
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());


    //auto t3 = std::chrono::high_resolution_clock::now();

    /*std::chrono::duration<double, std::milli> t2t1 = t2 - t1;
    std::chrono::duration<double, std::milli> t3t2 = t3 - t2;

    std::cout << t2t1.count() << "\n";
    std::cout << t3t2.count() << "\n";*/



    std::cout << "Trace finished...\n";

   
    /*const dim3 THREADS(8,8);
    const dim3 blocks(raster.getWidth()/THREADS.x+1, raster.getHeight()/THREADS.y+1);
    render<<<blocks, THREADS>>>(data, raster.getWidth(), raster.getHeight());*/

    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    raster.writeData(data);

    checkError(cudaFree(data));

    std::cout << "Finished \n";
    return 0;
}