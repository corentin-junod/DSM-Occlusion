#include "../bvh/BVH.cuh"

#include <curand_kernel.h>

class Tracer{
public:
    Tracer(Array2D<float>& data, const float pixelSize, const bool useGPU);
    ~Tracer();

    void init(const bool prinInfos);
    void trace(const unsigned int raysPerPoint);

private:
    const unsigned int width;
    const unsigned int height;
    const float pixelSize;
    const bool useGPU;


    BVH<float>* bvh = nullptr; 
    Array2D<float>& data;
    Point3<float>* const points;
    curandState*   const randomState;
};