#include "../bvh/BVH.cuh"

#include <curand_kernel.h>

class Tracer{
public:
    Tracer(Array2D<float>& data, const float pixelSize);
    ~Tracer();

    void init(const bool useGPU, const bool prinInfos);
    void trace(const bool useGPU, const unsigned int raysPerPoint);

private:
    const unsigned int width;
    const unsigned int height;
    const float pixelSize;
    bool useGPUInit   = false;
    bool useGPURender = false;
    
    BVH<float>*     bvh; 
    Array2D<float>& data;
    Point3<float>*  points;
    curandState*    randomState = nullptr;
};