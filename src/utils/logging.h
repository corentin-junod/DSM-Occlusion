#include <chrono>
#include <iostream>

const char* const DATE_FORMAT = "%d.%m.%Y %H:%M:%S";

inline std::ostream& cout() {
#ifdef _WIN32
    std::tm ltime;
    localtime_s(&ltime, &std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    return std::cout << "[" << std::put_time(&ltime, DATE_FORMAT) << "]  ";
#else
    time_t currentTime = time(nullptr);
    struct tm* localTime = localtime(&currentTime);
    char formattedTime[50];
    strftime(formattedTime, sizeof(formattedTime), DATE_FORMAT, localTime);
    return std::cout << "[" << formattedTime << "]  ";
#endif
}