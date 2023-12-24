#include <iostream>

#include "utils/utils.cuh"
#include "IO/Raster.h"

__global__ 
void render(float* const buffer, const int width, const int height) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>=width || j>=height) return;
    const int pixel_index = j*width + i;
    buffer[pixel_index] *= 20;
}

int main(){
    const char* filename = "data/data.tif";
    const char* outputFilename = "data/output.tif";
    Raster* raster = new Raster(filename, outputFilename);

    raster->printInfos();
    //printDevicesInfos();

    const dim3 THREADS(8,8);
    const int nbPixels = raster->getWidth() * raster->getHeight();

    //checkError(cudaMallocManaged(&buffer, nbPixels*sizeof(char)));

    //const dim3 blocks(raster.width/THREADS.x+1, raster.height/THREADS.y+1);
    //render<<<blocks, THREADS>>>(raster.data, raster.width, raster.height);

    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    float* data = raster->getData();

    for(int i=0; i<raster->getWidth(); i++){
        for(int j=0; j<raster->getHeight(); j++){
            //std::cout << data[i+j*raster->getWidth()] << " -- ";
            data[i+j*raster->getWidth()] *= -1;
            //std::cout << data[i+j*raster->getHeight()] << '\n';
        }
    }

    data[0] = 0;

    raster->writeData();
    //checkError(cudaFree(buffer));

    std::cout << "Finished \n";
    return 0;
}