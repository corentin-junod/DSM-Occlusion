#pragma once

#include <curand_kernel.h>
#include <random>

std::default_random_engine generator;
std::uniform_real_distribution<> rndDist = std::uniform_real_distribution<>(-1.0, 1.0);

template<typename T>
class Vec3{

public:
    __host__ __device__ Vec3(T x=0, T y=0, T z=0) : x(x), y(y), z(z){};

    __device__ static Vec3<T> randomInHemisphere(curandState_t& state){
        T x, y, z;
        do{
            x = curand_uniform(&state) * 2 - 1;
            y = curand_uniform(&state) * 2 - 1;
            z = curand_uniform(&state) * 2 - 1; 
        }while(x*x + y*y + z*z > 1);
        return Vec3(x,y,z<0?-z:z);
    };

    __host__ static Vec3<T> randomInHemisphere(){
        T x, y, z;
        do{
            x = rndDist(generator);
            y = rndDist(generator);
            z = rndDist(generator); 
        }while(x*x + y*y + z*z > 1);
        return Vec3(x,y,z<0?-z:z);
    };

    T x, y, z;
};