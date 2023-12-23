#include "utils/utils.cuh"
#include "IO/IO.hpp"

__global__ 
void render(unsigned char* const buffer, const int width, const int height, const int channels) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>=width || j>=height) return;
    const int pixel_index = j*width*channels + i*channels;
    buffer[pixel_index]   = 255 * i / width;
    buffer[pixel_index+1] = 0;
    buffer[pixel_index+2] = 0;
}

int main(){
    const char* filename = "data.tif";
    const char* outputFilename = "output.tif";
    Raster raster = copyAndGetRaster(filename, outputFilename);

    writeRaster(raster);


    return 0;
    //printDevicesInfos();

    const dim3 THREADS(8,8);

    const int width  = 800;
    const int height = 800;
    const int channels = 3;
    const int nbPixels = width * height;

    unsigned char* buffer;
    checkError(cudaMallocManaged(&buffer, nbPixels*channels*sizeof(char)));

    const dim3 blocks(width/THREADS.x+1,height/THREADS.y+1);
    render<<<blocks, THREADS>>>(buffer, width, height, channels);

    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    std::cout<<"P3\n"<<width<<" "<<height<<"\n255\n";
    for(int y=0; y<height; y++){
        for(int x=0; x<width; x++){
            size_t pixel_index = y*3*width + x*3;
            std::cout<<buffer[pixel_index]<<" "<<buffer[pixel_index+1]<<" "<<buffer[pixel_index+2]<<"\n";
        }
    }

    checkError(cudaFree(buffer));
    return 0;
}