#include "Tracer.cuh"

#include "../utils/utils.cuh"
#include "../array/Array.cuh"

#include <random>

std::default_random_engine genEngine;
std::uniform_real_distribution<> uniform0_1 = std::uniform_real_distribution<>(0.001, 1);

constexpr unsigned int NB_STRATIFIED_DIRS = 32;
constexpr unsigned int SEED = 1423;
constexpr dim3 threads(8,8);


__host__ __device__ 
void initRender(const Array2D<float>& data, Array2D<Point3<float>>& points, Array2D<Point3<float>*>& pointsPointers, BVH<float>& bvh, const float pixelSize){
    for(unsigned int y=0; y<data.height(); y++){
        for(unsigned int x=0; x<data.width(); x++){
            const unsigned int index = y*data.width()+x;
            points[index] = Point3<float>((float)x*pixelSize,(float)y*pixelSize, data[index]);
            pointsPointers[index] = &(points[index]);
        }
    }
    bvh.build(pointsPointers);
}

__global__
void initRenderGPU(const Array2D<float>& data, Array2D<Point3<float>>& points, Array2D<Point3<float>*>& pointsPointers, BVH<float>& bvh, const float pixelSize) {
    const unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;
    initRender(data, points, pointsPointers, bvh, pixelSize);
}


__host__ 
void render(Array2D<float>& data, const unsigned int index, const unsigned int raysPerPoint, Array2D<Point3<float>>& points, BVH<float>& bvh, BVHNode<float>** traceBuffer, const unsigned int traceBufferSize){
    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray        = Ray<float>(origin, direction);
    float result = 0;
    for(unsigned int i=0; i<raysPerPoint; i++){
        const unsigned int segmentNumber = i%NB_STRATIFIED_DIRS;
        const float rnd1 = uniform0_1(genEngine);
        const float rnd2 = uniform0_1(genEngine);
        const float cosThetaOverPdf = ray.getDirection().setRandomInHemisphereCosine( NB_STRATIFIED_DIRS, segmentNumber, rnd1, rnd2);
        result += cosThetaOverPdf*bvh.getLighting(ray, &traceBuffer[index*traceBufferSize]);
    }
    data[index] = result/(PI*raysPerPoint); // Diffuse BSDF : f = 1/PI
}

__global__
void renderGPU(Array2D<float>& data, Array2D<Point3<float>>& points, BVH<float>& bvh, const unsigned int raysPerPoint, curandState* const rndState, unsigned int bufferSize){
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x>=data.width() || y>=data.height()) return;
    const unsigned int index = y*data.width() + x;

    curandState localRndState = rndState[index];
    curand_init(SEED, index, 0, &localRndState);

    extern __shared__ BVHNode<float>* traceBuffer[];
    const unsigned int traceBufferOffset = bufferSize*(threadIdx.x+8*threadIdx.y);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    Point3<float> origin  = points[index];
    Vec3<float> direction = Vec3<float>(0,0,0);
    Ray<float> ray        = Ray<float>(origin, direction);

    float result = 0;
    for(unsigned int i=0; i<raysPerPoint; i++){
        const float rnd1 = curand_uniform(&localRndState);
        const float rnd2 = curand_uniform(&localRndState);
        const unsigned int segmentNumber = i%NB_STRATIFIED_DIRS;
        const float cosThetaOverPdf = ray.getDirection().setRandomInHemisphereCosine( NB_STRATIFIED_DIRS , segmentNumber, rnd1, rnd2);
        result += cosThetaOverPdf*bvh.getLighting(ray, &traceBuffer[traceBufferOffset]);
    }
    data[index] = (result/raysPerPoint)*(1/PI); // Diffuse BSDF
}


Tracer::Tracer(Array2D<float>& data, const float pixelSize): 
    data(data), width(data.width()), height(data.height()), pixelSize(pixelSize), 
    points(Array2D<Point3<float>>(width, height)),
    bvh(new BVH<float>(width*height)){}

Tracer::~Tracer(){
    cudaFree(randomState);
    bvh->freeAllMemory();
    free(bvh);
}

void Tracer::init(const bool useGPU, const bool prinInfos){
    useGPUInit = useGPU;
    randomState = (curandState*) allocGPU(width*height, sizeof(curandState)); // TODO this should also be initialized when on CPU
    Array2D<Point3<float>*> pointsPointers(data.width(), data.height());

    if(useGPU){
        BVH<float>* bvhGPU = bvh->toGPU();
        Array2D<Point3<float>*>* pointsPointersGPU = pointsPointers.toGPU();
        Array2D<float>* dataGPU = data.toGPU();
        Array2D<Point3<float>>* pointsGPU = points.toGPU();
        initRenderGPU<<<1,1>>>(*dataGPU, *pointsGPU, *pointsPointersGPU, *bvhGPU, pixelSize);
        syncGPU();
        points.fromGPU(pointsGPU);
        data.fromGPU(dataGPU);
        pointsPointers.fromGPU(pointsPointersGPU);
        bvh->fromGPU(bvhGPU);
    }else{
        initRender(data, points, pointsPointers, *bvh, pixelSize);
        if(prinInfos) bvh->printInfos();
    }
    bvh->freeAfterBuild();
}


void Tracer::trace(const bool useGPU, const unsigned int raysPerPoint){
    useGPURender = useGPU;
    const unsigned int traceBufferSizePerThread = std::log2(bvh->size())+1;

    if(useGPU){
        const dim3 blocks(width/threads.x+1, height/threads.y+1);

        Array2D<Point3<float>>* pointsGPU = points.toGPU();
        BVH<float>* bvhGPU = bvh->toGPU();
        Array2D<float>* dataGPU = data.toGPU();
        const unsigned int sharedMem = threads.x*threads.y*traceBufferSizePerThread*sizeof(BVHNode<float>*);
        std::cout << sharedMem << '\n';
        renderGPU<<<blocks, threads, sharedMem>>>(*dataGPU, *pointsGPU, *bvhGPU, raysPerPoint, randomState, traceBufferSizePerThread);
        syncGPU();
        data.fromGPU(dataGPU);
        bvh->fromGPU(bvhGPU);
        points.fromGPU(pointsGPU);
    }else{
        BVHNode<float>** traceBuffer = (BVHNode<float>**) allocMemory(width*height*traceBufferSizePerThread, sizeof(BVHNode<float>*), useGPU);

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
                std::cout << "Progress " << 100*nextProgress << "%\n";
                std::flush(std::cout);
                nextProgress += 0.1;
            }
        }
        freeMemory(traceBuffer, useGPU);
    }
}
