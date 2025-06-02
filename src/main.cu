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
    "[--ambientPower power] "
    "[--skyPower power] "
    "[--sunPower power] "
    "[--sunAzimuth angle (in degree, between 0 and 360)] "
    "[--sunElevation angle (in degree, between 0 and 90)] "
    "[--sunAngularDiam (in degree, between 0 and 180)] "
    "[-b bias] "
    "[--tiled]"
    "[--startTile tileID]";

bool strEqual(const char* const s1, const char* const s2){
    return std::strncmp(s1, s2, MAX_STR_SIZE) == 0;
}

uint strToUint(const char* const str, uint min=0, uint max=1.0E9F){
    uint value = std::strtoul(str, NULL, 10);
    if(value < min || value > max){
        logger::cout() << "Error : Parameter with value" << value << " must be between " << min << " and " << max << "\n";
        exit(EXIT_FAILURE);
    }
    return value;
}

float strToFloat(const char* const str, float min=0, float max=1.0E9F){
    float value = std::strtof(str, NULL);
    if(value < min || value > max){
        logger::cout() << "Error : Parameter with value" << value << " must be between " << min << " and " << max << "\n";
        exit(EXIT_FAILURE);
    }
    return value;
}

int main(int argc, char* argv[]){
    const char* inputFilename  = nullptr;
    const char* outputFilename = "output.tif";
    uint  tileSize             = 2000;
    uint  tileBuffer           = tileSize/3;
    bool  printInfos           = false;
    float exaggeration         = 1.0;
    bool tiledRender           = false;
    uint startTile             = 0;

    LightingParams lightParams = LightingParams();

    for(int i=1; i<argc; i++){
        if(strEqual(argv[i], "-i")){
            inputFilename = argv[++i];
        }else if(strEqual(argv[i], "-o")){
            outputFilename = argv[++i];
        }else if(strEqual(argv[i], "-r")){
            lightParams.raysPerPoint = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "-t")){
            tileSize = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "-b")){
            tileBuffer = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "-e")){
            exaggeration = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "-B")){
            lightParams.maxBounces = strToUint(argv[++i]);
        }else if(strEqual(argv[i], "--ambientPower")){
            lightParams.ambientPower = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "--skyPower")){
            lightParams.skyPower = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "--sunPower")){
            lightParams.sunPower = strToFloat(argv[++i]);
        }else if(strEqual(argv[i], "--sunAzimuth")){
            lightParams.sunAzimuth = strToFloat(argv[++i], 0.0, 360.0);
        }else if(strEqual(argv[i], "--sunElevation")){
            lightParams.sunElevation = strToFloat(argv[++i], 0.0, 90.0);
        }else if(strEqual(argv[i], "--sunAngularDiam")){
            lightParams.sunAngularDiam = strToFloat(argv[++i], 0.0, 180.0);
        }else if(strEqual(argv[i], "--bias")){
            lightParams.bias = strToFloat(argv[++i]);
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
                Pipeline pipeline = Pipeline(rasterIn, &rasterOut, lightParams, tileSize, tileBuffer, exaggeration, startTile);
                while(pipeline.step()){}
            }else{
                std::filesystem::create_directory("./output_tiles/");
                Pipeline pipeline = Pipeline(rasterIn, nullptr, lightParams, tileSize, tileBuffer, exaggeration, startTile);
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