#pragma once

typedef unsigned int uint;
typedef unsigned char byte;

constexpr uint MAX_STR_SIZE = 200;

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI  = 2*PI;

struct Extent {
    int xMin = 0;
    int xMax = 0;
    int yMin = 0;
    int yMax = 0;
};

struct LightingParams {
    uint maxBounces = 0;
    uint raysPerPoint = 1024;
    float bias = 1;
    float ambientPower = 0;
    float skyPower = 1;
    float sunPower = 0;
    float sunAzimuth = 0; 
    float sunElevation = 45;
    float sunAngularDiam = 11.4;
    float materialReflectance = 1.0;
};

// Timing helper functions
#ifdef TIMING_DEBUG
    #include <chrono>
    #define TIMED_INIT(name) \
        std::chrono::nanoseconds name = std::chrono::nanoseconds::zero(); \
        std::chrono::time_point<std::chrono::high_resolution_clock> name##_start;
    #define TIMED_PRINT(name) std::cout << #name << " : " << name.count()/1e9 << "s \n";
    #define TIMED_ACC(name, ...) \
        name##_start = std::chrono::high_resolution_clock::now(); \
        __VA_ARGS__ \
        name += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - name##_start);

#else
    #define TIMED_INIT(name) {}
    #define TIMED_ACC(name, ...) __VA_ARGS__
    #define TIMED_PRINT(name) {}

#endif