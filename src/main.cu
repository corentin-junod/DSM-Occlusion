#include <cstdlib>
#include <iostream>
#include <string>

#include "utils/utils.cuh"
#include "IO/Raster.h"
#include "tracer/Tracer.cuh"

const char* const USAGE = "Usage : -i inputFile -o outputFile [-r raysPerPixel] [-t tileSize] [-p pixelSizeInMeter]\n";

int main(int argc, char* argv[]){
    char* inputFilename      = nullptr;
    char* outputFilename     = nullptr;
    unsigned int rayPerPoint = 128;
    unsigned int tileSize    = 1000;
    float pixelSize          = 0.5;

    char opt;
    while ( (opt=getopt(argc, argv, "i:o:r:t:p:")) != -1) {
        switch (opt) {
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            case 'r':
                rayPerPoint = std::stoul(optarg);
                break;
            case 't':
                tileSize = std::stoul(optarg);
                break;
            case 'p':
                pixelSize = std::stof(optarg);
                break;
            default:
                std::cout << USAGE;
                exit(EXIT_FAILURE);
        }
    }

    if(inputFilename == nullptr || outputFilename == nullptr || pixelSize <= 0){
        std::cout << USAGE;
        exit(EXIT_FAILURE);
    }

    // TODO put these are parameters
    const bool PRINT_INFOS = true;
    const int TILE_BORDER = tileSize/3;

    Raster rasterIn  = Raster(inputFilename);
    Raster rasterOut = Raster(outputFilename, &rasterIn);

    if(PRINT_INFOS){
        rasterIn.printInfos();
        printDevicesInfos();     
    }

    const unsigned int nbTiles = std::ceil((float)rasterIn.getHeight()/tileSize) * std::ceil((float)rasterIn.getWidth()/tileSize);

    unsigned int nbTileProcessed = 0;
    for(int y=0; y<rasterIn.getHeight(); y+=tileSize){
        for(int x=0; x<rasterIn.getWidth(); x+=tileSize){
            const unsigned int width  = std::min(tileSize, rasterIn.getWidth()-x);
            const unsigned int height = std::min(tileSize, rasterIn.getHeight()-y);

            const unsigned int xMin = std::max(0, x-TILE_BORDER);
            const unsigned int yMin = std::max(0, y-TILE_BORDER);
            const unsigned int xMax = std::min(rasterIn.getWidth(), x+width+TILE_BORDER);
            const unsigned int yMax = std::min(rasterIn.getHeight(), y+height+TILE_BORDER);
            const unsigned int widthBorder  = xMax - xMin;
            const unsigned int heightBorder = yMax - yMin;

            std::cout << "Processing tile " << nbTileProcessed+1 << "/" << nbTiles << " (" <<  100*nbTileProcessed/nbTiles << "%)...\n";

            Array2D<float> data(widthBorder, heightBorder);
            rasterIn.readData(data.begin(), xMin, yMin, widthBorder, heightBorder);

            Array2D<Float> dataFloat(widthBorder, heightBorder);
            for(unsigned int i=0; i<data.size(); i++){
                dataFloat[i] = (Float)data[i];
            }

            Tracer tracer = Tracer(dataFloat, pixelSize);

            std::cout << "> Building BVH...\n";
            tracer.init(false, false);

            std::cout << "> Start tracing...\n";
            tracer.trace(true, rayPerPoint);

            std::cout << "> Writing result...\n";
            unsigned int i=0;
            unsigned int j=0;
            for(unsigned int curY=yMin; curY < yMax; curY++){
                for(unsigned int curX=xMin; curX < xMax; curX++){
                    if(curY >= y && curX >= x && curY < y+height && curX < x+width){
                        data[i++] = (float)dataFloat[j];
                    }
                    j++;
                }
            }

            rasterOut.writeData(data.begin(), x, y, width, height);
            nbTileProcessed++;
        }
    }

    std::cout << "Finished \n";
    return EXIT_SUCCESS;
}