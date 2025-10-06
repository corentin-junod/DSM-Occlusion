#pragma once

#include "../utils/definitions.cuh"
#include "../utils/logging.h"
#include "../IO/Raster.h"
#include "../array/Array.cuh"
#include "../tracer/Tracer.cuh"

#include <string>
#include <thread>
#include <chrono>

using namespace std;
using namespace logger;

typedef chrono::time_point<chrono::high_resolution_clock> timePoint;

constexpr int NB_PIPELINE_STAGES = 4;

const std::string STAGE_NAMES[4] = {
    "READ_DATA", "INIT_TILE", "TRACE", "WRITE_DATA"
};

struct PipelineState {
    const Array2D<float>* dataIn = nullptr;
    Array2D<float>* dataOut      = nullptr;
    Tracer* tracer               = nullptr;
    Extent extent;
    int id = -1;
    uint width  = 0;
    uint height = 0;
    int x = 0;
    int y = 0;
    bool hasData  = false;
    bool finished = false;
};

struct PipelineStage {
    PipelineState* state = nullptr;
    std::thread* thread  = nullptr;
    bool ready = false;
    int id = -1;
};

class Pipeline{
public:
    Pipeline(Raster& rasterIn, Raster* rasterOut, const LightingParams params, uint tileSize, uint tileBuffer, float exaggeration, uint startTile, GDALDataType outputType);
    ~Pipeline();
    bool step();

private:
    PipelineStage stages[NB_PIPELINE_STAGES]; 
    Raster& rasterIn;
    Raster* rasterOut;

    static void waitForNextStep(PipelineStage* stage, timePoint startTime);
    static void readData(PipelineStage* stage, const Raster* rasterIn, uint tileSize, uint tileBuffer, uint startTile);
    static void initTile(PipelineStage* stage, float pixelSize, float exaggeration=1.0);
    static void trace(PipelineStage* stage, const LightingParams lightParams);
    static void writeData(PipelineStage* stage, const Raster* const rasterOut, const Raster* rasterIn, GDALDataType outputType);
};