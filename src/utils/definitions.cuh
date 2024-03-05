#pragma once

typedef unsigned int uint;
typedef unsigned char byte;

constexpr uint MAX_STR_SIZE = 200;

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI  = 2*PI;
constexpr float ONE_OVER_PI = 1/PI;


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
    #define TIMED_INIT(...) {}
    #define TIMED_ACC(name, content) content;
    #define TIMED_PRINT(name) {}

#endif