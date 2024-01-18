#include <iostream>

#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "tracer/Tracer.cuh"

int main(){
    const bool PRINT_INFOS = true;
    const char* filename = "data/input.tif";
    const char* outputFilename = "data/output.tif";

    const unsigned int RAYS_PER_POINT = 128;
    const float pixelSize = 0.5;

    Raster raster = Raster(filename, outputFilename);

    if(PRINT_INFOS){
        raster.printInfos();
        printDevicesInfos();     
    }

    std::cout << "Reading data...\n";
    Array2D<float> data(raster.getWidth(), raster.getHeight());
    raster.readData(data.begin());

    Tracer tracer = Tracer(data, pixelSize);

    std::cout << "Building BVH...\n";
    tracer.init(false, true);
    std::cout << "BVH built\n";

    std::cout << "Start tracing...\n";
    tracer.trace(true, RAYS_PER_POINT);
    std::cout << "Tracing finished...\n";

    raster.writeData(data.begin());

    std::cout << "Finished \n";
    return 0;
}