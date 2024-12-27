#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "utils/definitions.cuh"
#include "utils/logging.h"
#include "IO/Raster.h"
#include "pipeline/Pipeline.cuh"

const char* const USAGE =
    "Usage : -i inputFile "
    "[-o outputFile] "
    "[-r raysPerPixel] "
    "[-t tileSize (in pixels)] "
    "[-b tileBuffer (in pixels)] "
    "[-B raysBounces] "
    "[-e exaggeration]"
    "[-b bias] "
    "[--tiled]"
    "[--startTile tileID]";

bool strEqual(const char* const s1, const char* const s2){
    return std::strncmp(s1, s2, MAX_STR_SIZE) == 0;
}

uint strToUint(const char* const str){
    return std::strtoul(str, NULL, 10);
}

float strToFloat(const char* const str){
    return std::strtof(str, NULL);
}

int main(int argc, char* argv[]){
    const char* inputFilename  = nullptr;
    const char* outputFilename = "output.tif";
    uint  rayPerPoint          = 512;
    uint  tileSize             = 2000;
    uint  tileBuffer           = tileSize/3;
    bool  printInfos           = false;
    float exaggeration         = 1.0;
    uint maxBounces            = 0;
    float bias                 = 1;
    bool tiledRender           = false;
    uint startTile             = 0;

    for(int i=1; i<argc; i++){
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
        }else if(strEqual(argv[i], "-e")){
            exaggeration = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "-B")){
            maxBounces = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "--bias")){
            bias = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "--info")){
            printInfos = true;
        }else if(strEqual(argv[i], "--tiled")){
            tiledRender = true;
        }else if(strEqual(argv[i], "--startTile")){
            startTile = strToUint(argv[++i]);
        }else{
            logger::cout() << "Error : Invalid argument : " << argv[i] << '\n' << USAGE;
            exit(EXIT_FAILURE);
        }
    }

    if(inputFilename == nullptr){
        logger::cout() << "Error : Input file required\n" << USAGE;
        exit(EXIT_FAILURE);
    }

    logger::cout() << "Creating output file ...\n";
    auto startTime = chrono::high_resolution_clock::now();
    {
        try {
            Raster rasterIn = Raster(inputFilename);

            if(printInfos){
                rasterIn.printInfos();
                printDevicesInfos();     
            }

            if(!tiledRender){
                Raster rasterOut = Raster(outputFilename, &rasterIn);
                Pipeline pipeline = Pipeline(rasterIn, &rasterOut, tileSize, rayPerPoint, tileBuffer, exaggeration, maxBounces, bias, startTile);
                while(pipeline.step()){}
            }else{
                std::filesystem::create_directory("./output_tiles/");
                Pipeline pipeline = Pipeline(rasterIn, nullptr, tileSize, rayPerPoint, tileBuffer, exaggeration, maxBounces, bias, startTile);
                while(pipeline.step()){}
            }
            logger::cout() << "Writing statistics and closing file... \n";
        } catch (const std::exception& e) {
            logger::cout() << "Error : " << e.what() << "\n";
            return EXIT_FAILURE;
        } catch(...){
            logger::cout() << "Unkown error occured\n";
            return EXIT_FAILURE;
        }
    }
    int elapsedTime = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - startTime).count();
    logger::cout() << "Finished in " << elapsedTime << "s\n";
    return EXIT_SUCCESS;
}