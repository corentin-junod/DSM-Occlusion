#pragma once

#include "../utils/definitions.cuh"
#include "cuda_fp16.h"
#include <iostream>

template<typename T>
class Vec3{

public:
    __host__ __device__ Vec3(T x=0, T y=0, T z=0) : x(x), y(y), z(z){};

    __host__ __device__ void normalize() {
        const T norm = sqrtf(x*x + y*y + z*z);
        x /= norm;
        y /= norm;
        z /= norm;
    }

    __host__ __device__ T getNormSquared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ T dot(const Vec3<T>& other) const {
        return x*other.x + y*other.y + z*other.z; 
    }

    __device__ __forceinline__ 
    Float setRandomInHemisphereCosineGPU(const Float nbSegments, const Float nbSamples, const unsigned char sampleId, const Float rndPhi, const Float rndTheta){
        const Float currentSampleSegment = floorf(sampleId* fdividef(nbSegments, nbSamples));
        const Float segmentSize = fdividef(TWO_PI, nbSegments);
        const Float phi = rndPhi * fdividef(TWO_PI,segmentSize) + segmentSize * currentSampleSegment;
        const Float theta = acosf(sqrtf(rndTheta));
        const Float sinTheta = sinf(theta);
        x = sinTheta*cosf(phi);
        y = sinTheta*sinf(phi);
        z = cosf(theta);
        return PI; // pdf = cosTheta / PI, so cosTheta / pdf = PI
    }

    __host__ 
    Float setRandomInHemisphereCosineHost(const unsigned char nbSegments, const unsigned char segmentNumber, const Float rndNumber1, const Float rndNumber2){
        const Float segmentSize = (Float)2*(Float)PI/(Float)nbSegments;
        const Float phi = rndNumber1 * (Float)2*(Float)PI / (Float)segmentSize + (Float)segmentSize * (Float)segmentNumber;
        const Float theta = acos(sqrt((float)rndNumber2));
        const Float sinTheta = sin((float)theta);
        const Float cosTheta = cos((float)theta);
        x = sinTheta*(Float)cos((float)phi);
        y = sinTheta*(Float)sin((float)phi);
        z = cosTheta;
        const Float pdf = cosTheta / PI;
        return cosTheta / pdf;
    }

    __host__ 
    Float setRandomInHemisphereUniform(const unsigned char nbSegments, const unsigned char segmentNumber, const Float rndNumber1, const Float rndNumber2){
        const Float segmentSize = (Float)2*(Float)PI/(Float)nbSegments;
        const Float phi = rndNumber1 * (Float)segmentSize + (Float)segmentSize * (Float)segmentNumber;
        const Float theta = (Float)acos((float)rndNumber2);
        const Float sinTheta = sin((float)theta);
        const Float cosTheta = cos((float)theta);
        x = sinTheta*(Float)cos((float)phi);
        y = sinTheta*(Float)sin((float)phi);
        z = cosTheta;
        const Float pdf = (Float)1/((Float)2*PI);
        return cosTheta / pdf;
    };

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Vec3<T>& p) {
    os << "Vec3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}