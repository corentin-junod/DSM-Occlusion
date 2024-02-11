#include "Tracer.cuh"

#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <iostream>
#include <random>

std::default_random_engine genEngine;
std::uniform_real_distribution<> uniform0_1 = std::uniform_real_distribution<>(0.001, 1); // Not starting at zero to avoid dividing by zero

constexpr unsigned char NB_STRATIFIED_DIRS = 64; // TODO properly compute this number so that the render is not biased
constexpr unsigned int SEED = 1423; // For reproducible runs, can be any value

constexpr dim3 threads(8,8);

/*__host__ 
void render(Array2D<Float>& data, const unsigned int index, const unsigned int raysPerPoint, Array2D<Point3<Float>>& points, BVH& bvh, int* traceBuffer, const unsigned int traceBufferSize){
    const Point3<Float> origin  = points[index];
    Vec3<Float> direction = Vec3<Float>(0,0,0);
    Float result = 0;
    for(unsigned int i=0; i<raysPerPoint; i++){
        const unsigned int segmentNumber = i%NB_STRATIFIED_DIRS;
        const Float rnd1 = uniform0_1(genEngine);
        const Float rnd2 = uniform0_1(genEngine);
        const Float cosThetaOverPdf = direction.setRandomInHemisphereCosineHost( NB_STRATIFIED_DIRS, segmentNumber, rnd1, rnd2);
        result += cosThetaOverPdf*bvh.getLighting(origin, direction, &traceBuffer[index*traceBufferSize]);
    }
    data[index] = result/(PI*(Float)raysPerPoint); // Diffuse BSDF : f = 1/PI
}*/

__global__
void renderGPU(const Array2D<Float>& data, const Array2D<Point3<Float>>& points, const BVH& bvh, const unsigned int raysPerPoint, curandState* const rndState){    
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;

    extern __shared__ Float sharedMem[];
    BVHNode* const cache = (BVHNode*) &sharedMem[threads.x*threads.y];

    for(int i=0; i<raysPerPoint/(threads.x*threads.y); i++){
        const int curIndex = (i+1)*(threadIdx.x + threadIdx.y * threads.x);
        sharedMem[curIndex] = (float)curIndex/raysPerPoint;
    }

    const unsigned int index = y*data.width() + x;

    curandState localRndState = rndState[index];
    curand_init(SEED, index, 0, &localRndState);

    const Point3<Float> origin(points[index].x, points[index].y, points[index].z);
    Vec3<Float> direction = Vec3<Float>(0, 0, 0);
    
    __syncthreads(); // Wait for each thread to initialize its part of the shared memory

    Float result = 0;
    for(unsigned short i=0; i<raysPerPoint; i++){
        const Float rndPhi   = fminf( sharedMem[i] + 0.005*curand_uniform(&localRndState), 1);
        const Float rndTheta = sharedMem[i+1];
        const Float cosThetaOverPdf = direction.setRandomInHemisphereCosineGPU(NB_STRATIFIED_DIRS, raysPerPoint, i, rndPhi, rndTheta);
        const Vec3<Float> invDir(fdividef(1,direction.x), fdividef(1,direction.y), fdividef(1,direction.z));
        
        result += cosThetaOverPdf*bvh.getLighting(origin, invDir);
    }
    data[index] = (result/raysPerPoint)*ONE_OVER_PI; // Diffuse BSDF
}

Tracer::Tracer(Array2D<Float>& data, const Float pixelSize): 
    data(data), width(data.width()), height(data.height()), pixelSize(pixelSize), 
    points(Array2D<Point3<Float>>(width, height)),
    bvh(new BVH(width*height)){}

Tracer::~Tracer(){
    cudaFree(randomState);
    bvh->freeAllMemory();
    free(bvh);
}

void Tracer::init(const bool prinInfos){
    randomState = (curandState*) allocGPU(width*height, sizeof(curandState));
    Array2D<Point3<Float>*> pointsPointers(data.width(), data.height());

    for(unsigned int y=0; y<data.height(); y++){
        for(unsigned int x=0; x<data.width(); x++){
            const unsigned int index = y*data.width()+x;
            points[index] = Point3<Float>((Float)x*pixelSize,(Float)y*pixelSize, data[index]);
            pointsPointers[index] = &(points[index]);
        }
    }
    bvh->build(pointsPointers);

    if(prinInfos){
        bvh->printInfos();
    }
    bvh->freeAfterBuild();
}

void Tracer::trace(const bool useGPU, const unsigned int raysPerPoint){
    const unsigned int traceBufferSizePerThread = std::log2(bvh->size());

    //if(useGPU){
        const dim3 blocks(width/threads.x+1, height/threads.y+1);

        Array2D<Point3<Float>>* pointsGPU = points.toGPU();
        BVH* bvhGPU = bvh->toGPU();
        Array2D<Float>* dataGPU = data.toGPU();
        const unsigned int sharedMem = (raysPerPoint+1)*sizeof(Float);//+64*sizeof(BVHNode); 
        renderGPU<<<blocks, threads, sharedMem>>>(*dataGPU, *pointsGPU, *bvhGPU, raysPerPoint, randomState);
        syncGPU();
        data.fromGPU(dataGPU);
        bvh->fromGPU(bvhGPU);
        points.fromGPU(pointsGPU);
    /*}else{
        int* traceBuffer = (int*) allocMemory(width*height*traceBufferSizePerThread, sizeof(int), useGPU);

        float progress = 0;
        float nextProgress = 0.1;

        #pragma omp parallel for
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                render(data, y*width+x, raysPerPoint, points, *bvh, traceBuffer, traceBufferSizePerThread);
            }

            #pragma omp atomic
            progress++;
            if(progress >= nextProgress*height){
                nextProgress += 0.1;
            }
        }
        freeMemory(traceBuffer, useGPU);
    }*/
}
