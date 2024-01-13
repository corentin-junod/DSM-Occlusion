#include "../bvh/BVH.cuh"

#include <curand_kernel.h>

class Tracer{

public:
    Tracer(float* const data, const bool useGPU, const unsigned int width, const unsigned int height, const float pixelSize);
    ~Tracer();

    void init(const bool prinInfos);
    void trace(const unsigned int raysPerPoint);

private:
    const unsigned int width;
    const unsigned int height;
    const float pixelSize;
    const bool useGPU;

    float* const data;
    Point3<float>* points;
    BVH<float> bvh; 
    curandState* randomState = nullptr;
};