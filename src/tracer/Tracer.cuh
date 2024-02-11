#include "../bvh/BVH.cuh"
#include <curand_kernel.h>

class Tracer{
public:
    Tracer(Array2D<Float>& data, const Float pixelSize);
    ~Tracer();

    void init(const bool prinInfos);
    void trace(const bool useGPU, const unsigned int raysPerPoint);

private:
    const unsigned int width;
    const unsigned int height;
    const Float pixelSize;
    bool useGPURender = false;
    
    BVH* bvh; 
    Array2D<Float>& data;
    Array2D<Point3<Float>> points;
    curandState* randomState = nullptr;
};