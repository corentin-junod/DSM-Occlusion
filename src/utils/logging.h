#include <chrono>
#include <iostream>
#include <iomanip>

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