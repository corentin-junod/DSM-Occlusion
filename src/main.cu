#include <iostream>

#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "sizedArray/SizedArray.cuh"
#include "primitives/Point3.cuh"

__global__ 
void render(float* const positions, const int width, const int height) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i>=width || j>=height) return;
    positions[j*width+i] *= 20;
}

__global__ 
void createPoints(float* const positions, const int width, const int height, SizedArray<Point3<float>> points){
    for(int x=0; x<width; x++){
        for(int y=0; y<height; y++){
            const int index = y*width+x;
            points[index] = Point3<float>(x,y,positions[index]);
        }
    }
}

/*
__global__ 
void buildBVH(float* const positions, const int width, const int height){
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    bvh = new BVH<float>(positions)
}

void trace(Ray ray, BVH bvh){
    Box* box = bvh.findIntersect(ray);
    if(box != nullptr){

    }
}
*/

int main(){
    const char* filename = "data/data.tif";
    const char* outputFilename = "data/output.tif";
    Raster raster = Raster(filename, outputFilename);

    //raster.printInfos();
    //printDevicesInfos();

    const unsigned int nbPixels = raster.getWidth()*raster.getHeight();

    float* data;
    checkError(cudaMallocManaged(&data, nbPixels*sizeof(float)));
    raster.readData(data);

    Point3<float>* pointsData;
    checkError(cudaMallocManaged(&pointsData, nbPixels*sizeof(Point3<float>)));
    SizedArray<Point3<float>> points(pointsData, nbPixels);
    createPoints<<<1,1>>>(data, raster.getWidth(), raster.getHeight(), points);


    const dim3 THREADS(8,8);
    const dim3 blocks(raster.getWidth()/THREADS.x+1, raster.getHeight()/THREADS.y+1);
    render<<<blocks, THREADS>>>(data, raster.getWidth(), raster.getHeight());

    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    raster.writeData(data);

    checkError(cudaFree(data));

    std::cout << "Finished \n";
    return 0;
}