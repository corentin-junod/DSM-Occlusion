#include <cstddef>
#include <iostream>
#include <cmath>

#include "utils/utils.cuh"
#include "utils/definitions.cuh"
#include "IO/Raster.h"
#include "tracer/Tracer.cuh"

int main(){
    const bool USE_GPU = false;
    const bool PRINT_INFOS = true;
    const char* filename = "data/input.tif";
    const char* outputFilename = "data/output.tif";

    const unsigned int RAYS_PER_POINT = 64;
    const float pixelSize = 0.5;


    Raster raster = Raster(filename, outputFilename);
    const unsigned long nbPixels = raster.getWidth()*raster.getHeight();

    if(PRINT_INFOS){
        raster.printInfos();
        printDevicesInfos();     
    }

    std::cout << "Reading data...\n";
    float* data;
    if(USE_GPU){
        data = (float*)allocGPU(nbPixels, sizeof(float));
    }else{
        data = (float*)calloc(nbPixels, sizeof(float));
    }
    raster.readData(data);

    Tracer tracer = Tracer(data, USE_GPU, raster.getWidth(), raster.getHeight(), pixelSize);

    std::cout << "Building BVH...\n";
    tracer.buildBVH(!USE_GPU);
    std::cout << "BVH built\n";

    std::cout << "Start tracing...\n";
    tracer.trace(RAYS_PER_POINT);
    std::cout << "Tracing finished...\n";

    raster.writeData(data); //TODO copy memory back to host if on device

    if(USE_GPU){
        cudaFree(data);
    }else{
        free(data);
    }

    std::cout << "Finished \n";
    return 0;
}