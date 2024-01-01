#pragma once

#include <curand_kernel.h>
#include <random>

std::default_random_engine generator;
std::uniform_real_distribution<> rndDist = std::uniform_real_distribution<>(-1.0, 1.0);

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

    __device__ void setRandomInHemisphere(curandState_t& state, const unsigned int quadrant){
        do{
            x = curand_uniform(&state) * 2 - 1;
            y = curand_uniform(&state) * 2 - 1;
            z = curand_uniform(&state) * 2 - 1; 
        }while(x*x + y*y + z*z > 1);

        if(z<0){
            z *= -1;
        }
        /*if( (x < 0 && (quadrant == 0 || quadrant == 3)) || (x > 0 && (quadrant == 1 || quadrant == 2)) ){
            x *= -1;
        }
        if( (y < 0 && (quadrant == 2 || quadrant == 3)) || (y > 0 && (quadrant == 0 || quadrant == 1)) ){
            y *= -1;
        }*/

    };

    __host__ void setRandomInHemisphere(const unsigned int quadrant){
        do{
            x = rndDist(generator);
            y = rndDist(generator);
            z = rndDist(generator); 
        }while(x*x + y*y + z*z > 1);

        if(z<0){
            z *= -1;
        }
        /*if( (x < 0 && (quadrant == 0 || quadrant == 3)) || (x > 0 && (quadrant == 1 || quadrant == 2)) ){
            x *= -1;
        }
        if( (y < 0 && (quadrant == 2 || quadrant == 3)) || (y > 0 && (quadrant == 0 || quadrant == 1)) ){
            y *= -1;
        }*/
    };

    T x, y, z;
};

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const Vec3<T>& p) {
    os << "Vec3("<<p.x<<","<<p.y<<","<<p.z<<")";
    return os;
}