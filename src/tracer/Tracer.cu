#include "Tracer.cuh"

#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include "device_launch_parameters.h"
#include <random>

std::default_random_engine genEngine;
std::uniform_real_distribution<> uniform0_1 = std::uniform_real_distribution<>(0.001, 1); // Not starting at zero to avoid dividing by zero

constexpr uint NB_SEGMENTS_DIR = 16;
constexpr uint SEED = 1423; // For reproducible runs, can be any value
constexpr uint BLOCK_DIM_SIZE = 8;

__global__
void renderGPU(const Array2D<float>& data, const Array2D<Point3<float>>& points, const BinaryTree& bvh, curandState* const rndState, const LightingParams params){    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;
    const uint index = y*data.width() + x;
    const int raysPerDir = params.raysPerPoint / NB_SEGMENTS_DIR;

    curandState localRndState = rndState[index];
    curand_init(SEED, index, 0, &localRndState);

    const Point3<float> origin(points[index].x, points[index].y, points[index].z);
    Vec3<float> direction = Vec3<float>(0.0, 0.0, 0.0);

    __syncthreads(); // Wait for each thread to initialize its part of the shared memory

    float result = 0;
    for(uint i=0; i<params.raysPerPoint; i++){
        const float rndForTheta = powf(fdividef((i%raysPerDir) + curand_uniform(&localRndState), raysPerDir), params.bias);
        const float rndForPhi   = fdividef((i/raysPerDir) + curand_uniform(&localRndState), NB_SEGMENTS_DIR);
        direction.setRandomInHemisphereCosine(rndForPhi, rndForTheta);
        result += bvh.getLighting(origin, direction, localRndState, params);
        __syncthreads();
    }
    data[index] = result/params.raysPerPoint;
}

Tracer::Tracer(Array2D<float>& data, const float pixelSize, const float exaggeration): 
    data(data), pixelSize(pixelSize), exaggeration(exaggeration),
    points(Array2D<Point3<float>>(data.width(), data.height())), 
    bvh(BinaryTree(data.width()*data.height(), pixelSize)){}

Tracer::~Tracer(){
    cudaFree(randomState);
    bvh.freeAllMemory();
}

void Tracer::init(const bool prinInfos){
    randomState = (curandState*) allocGPU(sizeof(curandState), data.width()*data.height());
    Array2D<Point3<float>*> pointsPointers(data.width(), data.height());

    for(uint y=0; y<data.height(); y++){
        for(uint x=0; x<data.width(); x++){
            const uint index = y*data.width()+x;
            points[index] = Point3<float>(x*pixelSize,y*pixelSize, data[index]*exaggeration);
            pointsPointers[index] = &(points[index]);
        }
    }
    
    bvh.build(pointsPointers);
    if(prinInfos) bvh.printInfos();
    bvh.freeAfterBuild();
}

void Tracer::trace(bool useGPU, const LightingParams params){
    if(useGPU){
        const dim3 blockDims(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
        const dim3 gridDims(data.width()/blockDims.x+1, data.height()/blockDims.y+1);

        Array2D<Point3<float>>* pointsGPU = points.toGPU();
        BinaryTree* bvhGPU = bvh.toGPU();
        Array2D<float>* dataGPU = data.toGPU();
        renderGPU<<<gridDims, blockDims>>>(*dataGPU, *pointsGPU, *bvhGPU, randomState, params);
        syncGPU();
        data.fromGPU(dataGPU);
        bvh.fromGPU(bvhGPU);
        points.fromGPU(pointsGPU);
    }else{
        /*
        const uint traceBufferSizePerThread = std::log2(bvh.size());
        int* traceBuffer = (int*) allocMemory(width*height*traceBufferSizePerThread, sizeof(int), useGPU);
        float progress = 0;
        float nextProgress = 0.1;
        #pragma omp parallel for
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                //render(data, y*width+x, raysPerPoint, points, *bvh, traceBuffer, traceBufferSizePerThread);
                const Point3<float> origin  = points[index];
                Vec3<float> direction = Vec3<float>(0,0,0);
                float result = 0;
                for(uint i=0; i<raysPerPoint; i++){
                    const uint segmentNumber = i%NB_STRATIFIED_DIRS;
                    const float rnd1 = uniform0_1(genEngine);
                    const float rnd2 = uniform0_1(genEngine);
                    const float cosThetaOverPdf = direction.setRandomInHemisphereCosineHost( NB_STRATIFIED_DIRS, segmentNumber, rnd1, rnd2);
                    result += cosThetaOverPdf*bvh.getLighting(origin, direction, &traceBuffer[index*traceBufferSize]);
                }
                data[index] = result/(PI*(float)raysPerPoint); // Diffuse BSDF : f = 1/PI
            }

            #pragma omp atomic
            progress++;
            if(progress >= nextProgress*height){
                nextProgress += 0.1;
            }
        }
        freeMemory(traceBuffer, useGPU);*/
    }
}
