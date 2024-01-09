#pragma once

#include "../utils/definitions.cuh"

#include <curand_kernel.h>
#include <iostream>
#include <random>

std::default_random_engine generator;
std::uniform_real_distribution<> uniform0_1  = std::uniform_real_distribution<>(0.001, 1);

template<typename T>
class Vec3{

public:
    __host__ __device__ Vec3(T x=0, T y=0, T z=0) : x(x), y(y), z(z){};

    __host__ __device__ void normalize() {
        const T norm = sqrt(x*x + y*y + z*z);
        x /= norm;
        y /= norm;
        z /= norm;
    }

    __host__ __device__ T getNormSquared() const{
        return x*x + y*y + z*z;
    }

    __host__ __device__ T dot(const Vec3<T>& other) const {
        return x*other.x + y*other.y + z*other.z; 
    }

    __device__ float setRandomInHemisphereCosine(curandState_t& state, const float nbSegments, const float segmentNumber){
        const T segmentSize = 2*PI/nbSegments;
        const T phi = curand_uniform(&state) * 2*PI / segmentSize + segmentSize * segmentNumber;

        const T theta = acos(sqrt(curand_uniform(&state)));
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta; 
        return 1/(2*PI);
    }

    __host__ float setRandomInHemisphereCosine(const float nbSegments, const float segmentNumber){
        const T segmentSize = 2*PI/nbSegments;
        const T phi = uniform0_1(generator) * 2*PI / segmentSize + segmentSize * segmentNumber;
        const T theta = acos(sqrt(uniform0_1(generator)));
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta;
        const T pdf = cosTheta / PI;
        return cosTheta / pdf;
    }

    __host__ float setRandomInHemisphereUniform(const float nbSegments, const float segmentNumber){
        const T segmentSize = 2*PI/nbSegments;
        const T phi = uniform0_1(generator) * segmentSize + segmentSize * segmentNumber;
        const T theta = acos(uniform0_1(generator));
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta;
        const T pdf = 1/(2*PI);
        return cosTheta / pdf;
    };

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Vec3<T>& p) {
    os << "Vec3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}