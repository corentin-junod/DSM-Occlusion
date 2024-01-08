#include <cstddef>
#include <iostream>
#include <cmath>

#include "primitives/Vec3.cuh"
#include "utils/utils.cuh"
#include "utils/definitions.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "array/Array.cuh"
#include "primitives/Point3.cuh"
#include "primitives/Ray.cuh"


/*__global__
void buildBVH(BVH<float>** bvh, unsigned int nbPixels){
    &bvh = new BVH<float>(*pointsArray, BVHNodeMemory, bboxMemory, elementsMemory);
    bvh->build();
}*/

__global__ 
void initRender(int maxX, int maxY, curandState* randomState) {
   const int x = threadIdx.x + blockIdx.x * blockDim.x;
   const int y = threadIdx.y + blockIdx.y * blockDim.y;
   if(x>=maxX || y>=maxY) return;
   const int index = y*maxX + x;
   curand_init(1423, index, 0, &randomState[index]);
}


__global__
void trace(float* data, Point3<float>* points, int maxX, int maxY, BVH<float>* bvh, const int raysPerPoint, curandState* randomState, BVHNode<float>** traceBuffer, int traceBufferSize, int nbDirs){
    /*const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=maxX || y>=maxY) return;
    const int index = y*maxX + x;

    curandState localRndState = randomState[index];

    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray        = Ray<float>(origin, direction);

    float result = 0;
    for(int i=0; i<raysPerPoint; i++){
        const float p = ray.getDirection().setRandomInHemisphereCosine(localRndState, nbDirs, i%nbDirs);
        result += bvh->getLighting(ray, &traceBuffer[index*traceBufferSize])/p;
    }
    data[index] = result/raysPerPoint;*/
}

int main(){
    const bool USE_GPU = false;
    const bool PRINT_INFOS = true;
    const char* filename = "data/input.tif";
    const char* outputFilename = "data/output.tif";

    Raster raster = Raster(filename, outputFilename);
    const unsigned int nbPixels = raster.getWidth()*raster.getHeight();

    if(PRINT_INFOS){
        raster.printInfos();
        printDevicesInfos();     
    }

    if(USE_GPU){

    }else{

    }

    // Read data from raster
    float* data;
    checkError(cudaMallocManaged(&data, nbPixels*sizeof(float)));
    raster.readData(data);

    // Create points in 3D
    Point3<float>* points = (Point3<float>*) allocGPU(nbPixels, sizeof(Point3<float>));

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


    std::cout << "Building BVH...\n";

    BVH<float> bvh = BVH<float>(nbPixels);
    bvh.build(*pointsArray);
    bvh.printInfos();
    
    std::cout << "BVH built\n";


    // Trace
    constexpr unsigned int RAYS_PER_POINT = 64;
    constexpr int NB_STRATIFIED_DIRS = 32;

    const int traceBufferSizePerThread = std::log2(bvh.size())+1;
    BVHNode<float>** traceBuffer = (BVHNode<float>**)allocGPU(nbPixels*traceBufferSizePerThread, sizeof(BVHNode<float>*));

    std::cout << "Start tracing...\n";

    if(USE_GPU){
        /*const dim3 threads(8,8);
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
            bvh, RAYS_PER_POINT, randomState, traceBuffer, traceBufferSizePerThread, NB_STRATIFIED_DIRS);
        checkError(cudaGetLastError());
        checkError(cudaDeviceSynchronize());*/

    }else{

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
                    const float cosThetaOverP = ray.getDirection().setRandomInHemisphereCosine( NB_STRATIFIED_DIRS , i%NB_STRATIFIED_DIRS );
                    result += cosThetaOverP*bvh.getLighting(ray, &traceBuffer[index*traceBufferSizePerThread]);
                }
                data[index] = result/RAYS_PER_POINT;
            }

            #pragma omp atomic
            progress++;

            if(progress >= nextProgress*raster.getHeight()){
                std::cout << "Progress " << 100*nextProgress << "%\n";
                std::flush(std::cout);
                nextProgress += 0.1;
            }
        }
    }

    std::cout << "Tracing finished...\n";

    raster.writeData(data);

    cudaFree(data);

    std::cout << "Finished \n";
    return 0;
}