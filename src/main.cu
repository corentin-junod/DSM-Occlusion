#include <cstdlib>
#include <cstring>

#include "utils/definitions.cuh"
#include "utils/logging.h"
#include "IO/Raster.h"
#include "tracer/Tracer.cuh"

const char* const USAGE = "Usage : -i inputFile [-o outputFile] [-r raysPerPixel] [-t tile size (in pixels)] [-b tile buffer (in pixels)] \n";

bool strEqual(const char* const s1, const char* const s2){
    return std::strncmp(s1, s2, MAX_STR_SIZE) == 0;
}

uint strToUint(const char* const str){
    return std::strtoul(str, NULL, 10);
}

int main(int argc, char* argv[]){
    const char* inputFilename  = nullptr;
    const char* outputFilename = "output.tif";
    uint  rayPerPoint          = 128;
    uint  tileSize             = 2000;
    uint  tileBuffer           = tileSize/3;
    bool  printInfos           = false;

    for(uint i=1; i<argc; i++){
        if(strEqual(argv[i], "-i")){
            inputFilename = argv[++i];
        }else if(strEqual(argv[i], "-o")){
            outputFilename = argv[++i];
        }else if(strEqual(argv[i], "-r")){
            rayPerPoint = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "-t")){
            tileSize = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "-b")){
            tileBuffer = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "--info")){
            printInfos = true;
        }else{
            cout() << "Error : Invalid argument : " << argv[i] << '\n' << USAGE;
            exit(EXIT_FAILURE);
        }
    }

    if(inputFilename == nullptr){
        cout() << "Error : Input file required\n" << USAGE;
        exit(EXIT_FAILURE);
    }

    cout() << "Creating output file ...\n";
    {
        Raster rasterIn  = Raster(inputFilename);
        Raster rasterOut = Raster(outputFilename, &rasterIn);

        if(printInfos){
            rasterIn.printInfos();
            printDevicesInfos();     
        }

        const float noDataValue = rasterIn.getNoDataValue();
        const uint nbTiles = (uint)std::ceil((float)rasterIn.getHeight()/tileSize) * (uint)std::ceil((float)rasterIn.getWidth()/tileSize);

        uint nbTileProcessed = 0;
        for(int y=0; y<rasterIn.getHeight(); y+=tileSize){
            for(int x=0; x<rasterIn.getWidth(); x+=tileSize){
                const uint width  = std::min(tileSize, rasterIn.getWidth()-x);
                const uint height = std::min(tileSize, rasterIn.getHeight()-y);
                const uint xMin   = std::max(0, x-(int)tileBuffer);
                const uint yMin   = std::max(0, y-(int)tileBuffer);
                const uint xMax   = std::min(rasterIn.getWidth(), x+width+tileBuffer);
                const uint yMax   = std::min(rasterIn.getHeight(),y+height+tileBuffer);

                cout() << "Processing tile " << nbTileProcessed+1<<"/"<<nbTiles << " ("<<100*nbTileProcessed/nbTiles<<"%)...\n";

                Array2D<float> dataIn(xMax-xMin, yMax-yMin);
                Array2D<float> dataOut(xMax-xMin, yMax-yMin);

                rasterIn.readData(dataIn.begin(), xMin, yMin, xMax-xMin, yMax-yMin);

                bool hasData = false;
                for(uint i=0; i<dataIn.size(); i++){
                    if(dataIn[i] != noDataValue){
                        hasData = true;
                    }
                    dataOut[i] = dataIn[i];
                }

                if(hasData){
                    Tracer tracer = Tracer(dataOut, rasterIn.getPixelSize());

                    cout() << "> Building BVH...\n";
                    tracer.init(false);
                    cout() << "> Start tracing...\n";
                    tracer.trace(true, rayPerPoint);

                    cout() << "> Writing result...\n";
                    Array2D<float> dataCropped(width, height);
                    uint i=0, j=0;
                    for(uint curY=yMin; curY < yMax; curY++){
                        for(uint curX=xMin; curX < xMax; curX++){
                            if(curY>=y && curY<y+height && curX>=x && curX<x+width){
                                dataCropped[i++] = (dataIn[j] == noDataValue ? noDataValue : dataOut[j]);
                            }
                            j++;
                        }
                    }

                    rasterOut.writeData(dataCropped.begin(), x, y, width, height);
                }else{
                    cout() << "> Tile skipped because it had no data \n";
                }
            
                nbTileProcessed++;
            }
        }
        cout() << "Writing statistics and closing file... \n";
    }
    cout() << "Finished \n";
    return EXIT_SUCCESS;
}