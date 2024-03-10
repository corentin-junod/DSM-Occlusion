#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <ostream>

const char* const DATE_FORMAT = "%d.%m.%Y %H:%M:%S";

inline std::ostream& cout() {
    using namespace std;
    const time_t now = chrono::system_clock::to_time_t(chrono::system_clock::now());
#ifdef _WIN32
    tm ltime;
    localtime_s(&ltime, &now);
    return std::cout << "[" << put_time(&ltime, DATE_FORMAT) << "]  ";
#else
    return std::cout << "[" << put_time(localtime(&now), DATE_FORMAT) << "]  ";
#endif
}

inline void print_atomic(const std::string str) {
    static std::mutex cout_atomic_mutex;
    const std::lock_guard<std::mutex> lock(cout_atomic_mutex);
    cout() << str;
}

#ifdef DEBUG_MESSAGES
    inline void debug_print(std::string str){print_atomic(str);}
#else
    inline void debug_print(std::string str){}
#endif