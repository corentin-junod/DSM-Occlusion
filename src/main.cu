#include <cstdlib>
#include <cstring>

#include "utils/definitions.cuh"
#include "utils/logging.h"
#include "IO/Raster.h"
#include "tracer/Tracer.cuh"
#include "pipeline/Pipeline.cuh"

const char* const USAGE =
    "Usage : "
    "-i inputFile "
    "[-o outputFile] "
    "[-r raysPerPixel] "
    "[-t tile size (in pixels)] "
    "[-b tile buffer (in pixels)] "
    "[-e exaggeration]\n";

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
    uint  rayPerPoint          = 128;
    uint  tileSize             = 300;
    uint  tileBuffer           = tileSize/3;
    bool  printInfos           = false;
    float exaggeration         = 1.0;

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
        }else if(strEqual(argv[i], "-e")){
            exaggeration = strToFloat(argv[++i]);
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
        try {
            Raster rasterIn  = Raster(inputFilename);
            Raster rasterOut = Raster(outputFilename, &rasterIn);

            if(printInfos){
                rasterIn.printInfos();
                printDevicesInfos();     
            }

            {
                Pipeline pipeline = Pipeline(rasterIn, rasterOut, tileSize, rayPerPoint, tileBuffer, exaggeration);
                while(pipeline.step()){}
            }
            cout() << "Writing statistics and closing file... \n";
        } catch (const std::exception& e) {
            cout() << "Error : " << e.what() << "\n";
            return EXIT_FAILURE;
        } catch(...){
            cout() << "Unkown error occured\n";
            return EXIT_FAILURE;
        }
    }
    cout() << "Finished \n";
    return EXIT_SUCCESS;
}