#include <iostream>

#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "bvh/BVH.cuh"
#include "sizedArray/SizedArray.cuh"
#include "primitives/Point3.cuh"
#include "primitives/Ray.cuh"

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

__global__
void buildBVH(){
    
}

int main(){
    const char* filename = "data/data.tif";
    const char* outputFilename = "data/output.tif";
    Raster raster = Raster(filename, outputFilename);

    //raster.printInfos();
    //printDevicesInfos();

    const unsigned int nbPixels = raster.getWidth()*raster.getHeight();

    // Read data from raster
    float* data;
    checkError(cudaMallocManaged(&data, nbPixels*sizeof(float)));
    raster.readData(data);

    // Create points in 3D
    Point3<float>* pointsData;
    checkError(cudaMallocManaged(&pointsData, nbPixels*sizeof(Point3<float>)));
    SizedArray<Point3<float>> points(pointsData, nbPixels);
    //createPoints<<<1,1>>>(data, raster.getWidth(), raster.getHeight(), points);
    for(int x=0; x<raster.getWidth(); x++){
        for(int y=0; y<raster.getHeight(); y++){
            const int index = y*raster.getWidth()+x;
            points[index] = Point3<float>(x,y,data[index]);
        }
    }


    // Build BVH
    std::cout << "Building BVH...\n";
    BVH<float>* bvh = new BVH<float>(points);
    std::cout << "BVH built\n";

    // Trace
    std::cout << "Start tracing...\n";
    //curandState randState = curandState();
    //curand_init(4132, 0, 0, &randState);
    for(int i=0; i<points.getSize(); i++){
        Point3<float> point = points[i];

        float result = 0;
        for(int j=0; j<5; j++){
            Vec3<float> direction = Vec3<float>::randomInHemisphere();
            Ray<float> ray = Ray<float>(point, direction);
            result += bvh->isIntersecting(ray)?0.0:1.0;
        }
        
        data[i] = result/5;
        //std::cout << "Intersect : " << result << '\n';
    }
    std::cout << "Trace finished...\n";

   
    /*const dim3 THREADS(8,8);
    const dim3 blocks(raster.getWidth()/THREADS.x+1, raster.getHeight()/THREADS.y+1);
    render<<<blocks, THREADS>>>(data, raster.getWidth(), raster.getHeight());*/

    checkError(cudaGetLastError());
    checkError(cudaDeviceSynchronize());

    raster.writeData(data);

    checkError(cudaFree(data));

    std::cout << "Finished \n";
    return 0;
}