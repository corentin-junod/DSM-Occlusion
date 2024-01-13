#include "Tracer.cuh"
#include "../utils/utils.cuh"

#include <random>

std::default_random_engine genEngine;
std::uniform_real_distribution<> uniform0_1 = std::uniform_real_distribution<>(0.001, 1);

constexpr unsigned int NB_STRATIFIED_DIRS = 32;
constexpr unsigned int SEED = 1423;
constexpr dim3 threads(8,8);


__host__ __device__ 
void initRender(float* data, Point3<float>* points, Point3<float>** pointsPointers, BVH<float>& bvh, float pixelSize, const unsigned int width, const unsigned int height){
    for(unsigned int y=0; y<height; y++){
        for(unsigned int x=0; x<width; x++){
            const int index = y*width+x;
            points[index] = Point3<float>((float)x*pixelSize,(float)y*pixelSize, data[index]);
            pointsPointers[index] = &(points[index]);
        }
    }
    Array<Point3<float>*> pointsPointersArray = Array<Point3<float>*>(pointsPointers, width*height);
    bvh.build(pointsPointersArray);
}

__global__
void initRenderGPU(float* data, Point3<float>* points, Point3<float>** pointsPointers, BVH<float> bvh, float pixelSize, const unsigned int width, const unsigned int height, curandState* const randomState) {
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=width || y>=height) return;
    const unsigned int index = y*width + x;
    curand_init(SEED, index, 0, &randomState[index]);
    initRender(data, points, pointsPointers, bvh, pixelSize, width, height);
}


__host__ 
void render(
    float* const data, const unsigned int index, const unsigned int raysPerPoint, 
    Point3<float>* points, BVH<float>* bvh, BVHNode<float>** traceBuffer, 
    const unsigned int traceBufferSizePerThread){
    
    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray        = Ray<float>(origin, direction);

    float result = 0;
    for(unsigned int i=0; i<raysPerPoint; i++){
        const unsigned int segmentNumber = i%NB_STRATIFIED_DIRS;
        const float rnd1 = uniform0_1(genEngine);
        const float rnd2 = uniform0_1(genEngine);
        const float cosThetaOverPdf = ray.getDirection().setRandomInHemisphereCosine( NB_STRATIFIED_DIRS , segmentNumber, rnd1, rnd2);
        result += cosThetaOverPdf*bvh->getLighting(ray, &traceBuffer[index*traceBufferSizePerThread]);
    }
    data[index] = (result/raysPerPoint)*(1/PI); // Diffuse BSDF
}

__global__
void renderGPU(float* data, Point3<float>* points, int width, int height, BVH<float> bvh, const int raysPerPoint, curandState* randomState, BVHNode<float>** traceBuffer, int traceBufferSize){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=width || y>=height) return;

    const unsigned int index = y*width + x;
    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray        = Ray<float>(origin, direction);

    float result = 0;
    for(unsigned int i=0; i<raysPerPoint; i++){
        const float rnd1 = curand_uniform(randomState);
        const float rnd2 = curand_uniform(randomState);
        const unsigned int segmentNumber = i%NB_STRATIFIED_DIRS;
        const float cosThetaOverPdf = ray.getDirection().setRandomInHemisphereCosine( NB_STRATIFIED_DIRS , segmentNumber, rnd1, rnd2);
        result += cosThetaOverPdf*bvh.getLighting(ray, &traceBuffer[index*traceBufferSize]);
    }
    data[index] = (result/raysPerPoint)*(1/PI); // Diffuse BSDF
}


Tracer::Tracer(float* const data, const bool useGPU, const unsigned int width, const unsigned int height, const float pixelSize) : 
    data(data), useGPU(useGPU), width(width), height(height), pixelSize(pixelSize), bvh(BVH<float>(useGPU, width*height)){
    if(useGPU) randomState = (curandState*) allocGPU(width*height, sizeof(curandState));
    points = (Point3<float>*) allocMemory(width*height, sizeof(Point3<float>), useGPU);
}

Tracer::~Tracer(){
    if(useGPU) cudaFree(randomState);
    freeMemory(points, useGPU);
}

void Tracer::init(const bool prinInfos){
    Point3<float>** pointsArray = (Point3<float>**) allocMemory(width*height, sizeof(Point3<float>*), useGPU); //TODO free after use

    if(useGPU){
        initRenderGPU<<<1,1>>>(data, points, pointsArray, bvh, pixelSize, width, height, randomState);
        checkError(cudaGetLastError());
        checkError(cudaDeviceSynchronize());
    }else{
        initRender(data, points, pointsArray, bvh, pixelSize, width, height);
        if(prinInfos) bvh.printInfos();
    }
    bvh.freeMemoryAfterBuild();
}


void Tracer::trace(const unsigned int raysPerPoint){
    const int traceBufferSizePerThread = std::log2(bvh.size())+1;
    BVHNode<float>** traceBuffer = (BVHNode<float>**) allocMemory(width*height*traceBufferSizePerThread, sizeof(BVHNode<float>*), useGPU);

    if(useGPU){
        const dim3 blocks(width/threads.x+1, height/threads.y+1);
        renderGPU<<<blocks, threads>>>(data, points, width, height, bvh, raysPerPoint, randomState, traceBuffer, traceBufferSizePerThread);
        checkError(cudaGetLastError());
        checkError(cudaDeviceSynchronize());
    }else{
        float progress = 0;
        float nextProgress = 0.1;

        #pragma omp parallel for
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                render(data, y*width+x, raysPerPoint, points, &bvh, traceBuffer, traceBufferSizePerThread);
            }

            #pragma omp atomic
            progress++;
            if(progress >= nextProgress*height){
                std::cout << "Progress " << 100*nextProgress << "%\n";
                std::flush(std::cout);
                nextProgress += 0.1;
            }
        }
    }
    freeMemory(traceBuffer, useGPU);
}
