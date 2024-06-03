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
void renderGpu(const Array2D<float>& data, const Array2D<Point3<float>>& points, const BVH& bvh, const uint raysPerPoint, const uint maxBounces, curandState* const rndState, const float bias){    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;
    const uint index = y*data.width() + x;
    const int raysPerDir = raysPerPoint / NB_SEGMENTS_DIR;

    curandState localRndState = rndState[index];
    curand_init(SEED, index, 0, &localRndState);

    const Point3<float> origin(points[index].x, points[index].y, points[index].z);
    Vec3<float> direction = Vec3<float>(0.0, 0.0, 0.0);
    
    __syncthreads(); // Wait for each thread to initialize its part of the shared memory

    float result = 0;
    for(uint i=0; i<raysPerPoint; i++){
        const float rndTheta = powf(fdividef((i%raysPerDir) + curand_uniform(&localRndState), raysPerDir), bias);
        const float rndPhi   = fdividef((i/raysPerDir) + curand_uniform(&localRndState), NB_SEGMENTS_DIR);

        direction.setRandomInHemisphereCosine(rndPhi, rndTheta);
        result += bvh.getLighting(origin, direction, &localRndState, maxBounces);
        __syncthreads();
    }
    data[index] = result/raysPerPoint;
}

__global__
void renderGpuShadowMap(const Array3D<byte>& data, const Array2D<Point3<float>>& points, const BVH& bvh, const byte rays_per_dir, const uint nb_dirs){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;
    const uint index = y*data.width() + x;

    Vec3<float> direction = Vec3<float>(0.0, 0.0, 0.0);
    const Point3<float> origin(points[index].x, points[index].y, points[index].z);

    for(uint dir=0; dir<nb_dirs; dir++){
        byte elevation = 1;
        for(; elevation<rays_per_dir; elevation++){
            const float phi = dir*(TWO_PI/nb_dirs);
            const float theta = (PI / 2.0)-elevation*((PI/2.0)/rays_per_dir); //TODO We can improve that by considering 0° is always occluded and 90° is always not occluded

            direction.setFromAngles(phi, theta);
            if (bvh.getLighting(origin, direction) > 0) {
                break;
            }
        }
        data.at(x, y, dir) = (elevation == rays_per_dir-1 ? elevation : elevation-1);
    }
}

Tracer::Tracer(const Array2D<float>& inputData, const float pixelSize, const float exaggeration, const uint maxBounces): 
    inputData(inputData), pixelSize(pixelSize), exaggeration(exaggeration), maxBounces(maxBounces),
    points(Array2D<Point3<float>>(inputData.width(), inputData.height())), 
    bvh(BVH(inputData.width()*inputData.height(), pixelSize)){}

Tracer::~Tracer(){
    cudaFree(randomState);
    bvh.freeAllMemory();
}

void Tracer::init(const bool prinInfos){
    randomState = (curandState*) allocGPU(sizeof(curandState), inputData.width()*inputData.height());
    Array2D<Point3<float>*> pointsPointers(inputData.width(), inputData.height());
    for(uint y=0; y<inputData.height(); y++){
        for(uint x=0; x<inputData.width(); x++){
            const uint index = y*inputData.width()+x;
            points[index] = Point3<float>(x*pixelSize,y*pixelSize, inputData[index]*exaggeration);
            pointsPointers[index] = &(points[index]);
        }
    }
    bvh.build(pointsPointers);
    if(prinInfos) bvh.printInfos();
    bvh.freeAfterBuild();
}

void Tracer::trace(Array2D<float>& outputData, const bool useGPU, const uint raysPerPoint, const float bias){
    if(useGPU){
        const dim3 blockDims(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
        const dim3 gridDims(outputData.width()/blockDims.x+1, outputData.height()/blockDims.y+1);

        Array2D<Point3<float>>* pointsGPU = points.toGPU();
        BVH* bvhGPU = bvh.toGPU();
        Array2D<float>* dataGPU = outputData.toGPU();
        renderGpu<<<gridDims, blockDims>>>(*dataGPU, *pointsGPU, *bvhGPU, raysPerPoint, maxBounces, randomState, bias);
        syncGPU();
        outputData.fromGPU(dataGPU);
        bvh.fromGPU(bvhGPU);
        points.fromGPU(pointsGPU);
    }
}

void Tracer::traceShadowMap(Array3D<byte>& outputData, const bool useGPU, const uint rays_per_dir, const uint nb_dirs){
    if(useGPU){
        const dim3 blockDims(BLOCK_DIM_SIZE, BLOCK_DIM_SIZE);
        const dim3 gridDims(outputData.width()/blockDims.x+1, outputData.height()/blockDims.y+1);

        Array2D<Point3<float>>* pointsGPU = points.toGPU();
        BVH* bvhGPU = bvh.toGPU();
        Array3D<byte>* dataGPU = outputData.toGPU();
        renderGpuShadowMap<<<gridDims, blockDims>>>(*dataGPU, *pointsGPU, *bvhGPU, (byte)rays_per_dir, nb_dirs);
        syncGPU();
        outputData.fromGPU(dataGPU);
        bvh.fromGPU(bvhGPU);
        points.fromGPU(pointsGPU);
    }
}

