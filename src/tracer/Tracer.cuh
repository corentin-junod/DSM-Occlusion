#pragma once

#include "../bvh/BVH.cuh"
#include <curand_kernel.h>

class Tracer{
public:
    Tracer(const Array2D<float>& inputData, const float pixelSize, const float exaggeration=1.0, const uint maxBounce=0);
    ~Tracer();

    void init(const bool prinInfos);
    void trace(Array2D<float>& outputData, const bool useGPU, const uint raysPerPoint, const float bias);
    void traceShadowMap(Array3D<byte>& outputData, const bool useGPU, const uint rays_per_dir, const uint nb_dirs);

private:
    const float pixelSize;
    const float exaggeration;
    const uint maxBounces;
    
    BVH bvh;
    const Array2D<float>& inputData;
    Array2D<Point3<float>> points;
    curandState* randomState = nullptr;
};