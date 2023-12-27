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

/*__global__ 
void render(float* const positions, const int width, const int height) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>=width || j>=height) return;
    positions[j*width+i] *= 20;
}*/

__global__ void initRender(int maxX, int maxY, curandState* randomState) {
   const int x = threadIdx.x + blockIdx.x * blockDim.x;
   const int y = threadIdx.y + blockIdx.y * blockDim.y;
   if(x>=maxX || y>=maxY) return;
   const int index = y*maxX + x;
   curand_init(1423, index, 0, &randomState[index]);
}

__global__
void trace(float* data, Array<Point3<float>>& pointsArray, int maxX, int maxY, BVH<float>& bvh, BVHNode<float>* bvhRoot, int rayPerPoint, curandState* randomState, BVHNode<float>** workingBuffer){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=maxX || y>=maxY) return;
    const int index = y*maxX + x;

    Point3<float> origin = Point3<float>(0,0,0);
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray = Ray<float>(origin, direction);


    ray.setOrigin(pointsArray[index]);
    float result = 0;
    for(int i=0; i<rayPerPoint; i++){
        ray.setDirection(Vec3<float>::randomInHemisphere(randomState));
        result += bvh.isIntersectingRec(ray, bvhRoot)?0.0:1.0;
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
    Array<Point3<float>> pointsArray = Array<Point3<float>>(points, nbPixels);
    for(int y=0; y<raster.getHeight(); y++){
        for(int x=0; x<raster.getWidth(); x++){
            const int index = y*raster.getWidth()+x;
            pointsArray[index] = Point3<float>(x,y,data[index]);
        }
    }

    // Build BVH
    std::cout << "Building BVH...\n";
    BVH<float> bvh = BVH<float>(pointsArray);
    // Copy memory to GPU
    BVHNode<float>* bvhNodes;
    checkError(cudaMallocManaged(&bvhNodes, bvh.size()*sizeof(BVHNode<float>)));
    bvh.copyNodes(bvhNodes);



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

    BVHNode<float>** workingBuffer;
    checkError(cudaMallocManaged(&workingBuffer, bvh.size()*sizeof(BVHNode<float>*))); 

    const dim3 threads(8,8);
    const dim3 blocks(raster.getWidth()/threads.x+1, raster.getHeight()/threads.y+1);
    curandState* randomState;
    checkError(cudaMalloc((void **)& randomState, nbPixels*sizeof(curandState)));
    initRender<<<blocks, threads>>>(raster.getWidth(), raster.getHeight(), randomState);
    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());
    trace<<<blocks, threads>>>(
        data, pointsArray, raster.getWidth(), raster.getHeight(),  bvh, bvhNodes, NB_RAY_PER_POINT, randomState, workingBuffer);
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