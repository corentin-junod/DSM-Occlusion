#pragma once

#include <curand_kernel.h>
#include <iostream>
#include <random>

#define PI 3.14159265358979323846

std::default_random_engine generator;
std::uniform_real_distribution<> uniform0_1  = std::uniform_real_distribution<>(0, 1);

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

        const T theta = acos(sqrt(curand_uniform(&state)*0.99 + 0.01));
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta; 
        return cosTheta/(2*PI);
    }

    __host__ float setRandomInHemisphereCosine(const float nbSegments, const float segmentNumber){
        const T segmentSize = 2*PI/nbSegments;
        const T phi = uniform0_1(generator) * 2*PI / segmentSize + segmentSize * segmentNumber;

        const T theta = acos(sqrt(uniform0_1(generator)*0.99 + 0.01));
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta; 
        return cosTheta/(2*PI);
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
        return 1/(2*PI);
    };

    /*__host__ float setRandomInHemisphereImportance(const float nbSegments, const float segmentNumber){
        const T segmentSize = 2*PI/nbSegments;
        const T phi = uniform0_1(generator) * 2*PI / segmentSize + segmentSize * segmentNumber;

        T p, q, theta;
        if(uniform0_1(generator) > 1.0/4.0){
            theta = acos(uniform0_1(generator)/2.0 + 0.5);
            p =(1.0/PI) * (3.0/4.0);
        }else{
            theta = acos(uniform0_1(generator)/2.0);
            p = (1.0/PI) * (1.0/4.0);
        }
        const T sinTheta = sin(theta);
        const T cosTheta = cos(theta);
        x = sinTheta*cos(phi);
        y = sinTheta*sin(phi);
        z = cosTheta; 
        
        return p;
    };
    
        __device__ void setRandomInHemisphere(curandState_t& state, const unsigned int quadrant){
        do{
            x = curand_uniform(&state) * 2 - 1;
            y = curand_uniform(&state) * 2 - 1;
            z = curand_uniform(&state) * 2 - 1; 
        }while(x*x + y*y + z*z > 1);

        if(z<0){
            z *= -1;
        }
        if( (x < 0 && (quadrant == 0 || quadrant == 3)) || (x > 0 && (quadrant == 1 || quadrant == 2)) ){
            x *= -1;
        }
        if( (y < 0 && (quadrant == 2 || quadrant == 3)) || (y > 0 && (quadrant == 0 || quadrant == 1)) ){
            y *= -1;
        }

    }
    
    */

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Vec3<T>& p) {
    os << "Vec3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}