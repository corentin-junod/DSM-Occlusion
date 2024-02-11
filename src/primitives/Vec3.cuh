#pragma once

#include "../utils/definitions.cuh"
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
    float setRandomInHemisphereCosineGPU(const float nbSegments, const float nbSamples, const byte sampleId, const float rndPhi, const float rndTheta){
        const float currentSampleSegment = floorf(sampleId* fdividef(nbSegments, nbSamples));
        const float segmentSize = fdividef(TWO_PI, nbSegments);
        const float phi = rndPhi * fdividef(TWO_PI,segmentSize) + segmentSize * currentSampleSegment;
        const float theta = acosf(sqrtf(rndTheta));
        const float sinTheta = sinf(theta);
        x = sinTheta*cosf(phi);
        y = sinTheta*sinf(phi);
        z = cosf(theta);
        return PI; // pdf = cosTheta / PI, so cosTheta / pdf = PI
    }

    __host__ 
    float setRandomInHemisphereCosineHost(const byte nbSegments, const byte segmentNumber, const float rndNumber1, const float rndNumber2){
        const float segmentSize = 2.0*PI/nbSegments;
        const float phi = rndNumber1 * TWO_PI / segmentSize + segmentSize * segmentNumber;
        const float theta = acos(sqrt(rndNumber2));
        const float sinTheta = sin(theta);
        const float cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta;
        const float pdf = cosTheta / PI;
        return cosTheta / pdf;
    }

    __host__ 
    float setRandomInHemisphereUniform(const byte nbSegments, const byte segmentNumber, const float rndNumber1, const float rndNumber2){
        const float segmentSize = 2.0*PI/nbSegments;
        const float phi = rndNumber1 * segmentSize + segmentSize * segmentNumber;
        const float theta = acos(rndNumber2);
        const float sinTheta = sin(theta);
        const float cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta;
        const float pdf = 1.0/TWO_PI;
        return cosTheta / pdf;
    };

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Vec3<T>& p) {
    os << "Vec3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}